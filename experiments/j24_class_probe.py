"""J24 (1), the question asked before any code: does the row128 eye carry an
object-CLASS signal that closeness bins "per object class" could be built from
"from the row128 rays AND NOTHING ELSE"?

    PYTHONPATH=. /data/venvs/jack-std/bin/python -m experiments.j24_class_probe

Not a learner, not a ledger spec: it builds worlds and reads names. The ray
row is `[distance/8 m, texture]` per ray (`fast_sim._row_vision`); texture is
`w0.py:225`: `_tex[gid] = ((gid * 0.37) % 0.8) + 0.1` — a hash of the geom's
ID. This prints, for the pilot's world (`pilot_b2.make_world`, body eye, body
motor, food_floor_only) on J24's seeds 941-943 and 20 more:
  (a) each floor food's texture value in each world;
  (b) in how many worlds a value that is FOOD in one world is a NON-food geom
      in another (so no fixed value-set means "food");
  (c) within one world, how many distinct values the foods carry and how many
      non-food geoms a ray can hit (so no single in-world value means "food").
The result is in docs/ORDERS.md INBOX (11-25) — see the RESULT block below.

RESULT (2026-11-25 07:1x UTC, box, jack-std, mujoco 3.17.0, nice 19):
 23 worlds (941-943, 281-300). NO CLASS SIGNAL. Every world has 3 floor foods
 (obj0-2) carrying 3 DIFFERENT texture values among the 29-34 other geoms a
 ray can hit, each with its own value. The values move with the world's geom
 count (36-41): 4 distinct food value-sets in 23 worlds, and J24's own three
 seeds all differ — 941 (.49 .86 .43), 942 (.86 .43 .80), 943 (.43 .80 .37).
 In 23/23 worlds a value that is food in another world is a NON-food geom
 here. 6 of the 41 values in use are food somewhere. So the channel is a
 per-OBJECT id, not a class: "food bins" cannot be computed from the row
 without reading which geom ids are food — simulator state no ray returned.
 J24 (1)'s stop clause applies: no `eye="bins16"` built, no learner seed.
"""
import collections

import mujoco
import numpy as np

SEEDS = (941, 942, 943) + tuple(range(281, 301))


def world_table(seed: int) -> dict:
    from experiments.pilot_b2 import PILOT_RULE, make_world
    e = make_world(seed, food_floor_only=PILOT_RULE["food_floor_only"],
                   eye_frame=PILOT_RULE["eye_frame"],
                   motor_frame=PILOT_RULE["motor_frame"],
                   native_substep=PILOT_RULE["native_substep"], eye="row128")
    e.reset(seed=seed)
    w, m = e.w, e.w.model
    foods = {n: int(g) for n, (g, _) in w.drives._food.items()
             if n in getattr(e, "floor_foods", ())}
    names = {g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}")
             for g in range(m.ngeom)}
    seen = [g for g in range(m.ngeom) if m.geom_group[g] != 3]   # not his body
    tex = {g: round(float(w._tex[g]), 4) for g in range(m.ngeom)}
    return {"seed": seed, "ngeom": int(m.ngeom), "foods": foods, "names": names,
            "tex": tex, "seen": seen}


def main() -> None:
    tabs = [world_table(s) for s in SEEDS]
    food_vals = collections.defaultdict(set)      # value -> worlds where food
    for t in tabs:
        fv = sorted({t["tex"][g] for g in t["foods"].values()})
        print(f"s{t['seed']} ngeom {t['ngeom']} foods "
              + " ".join(f"{n}={t['tex'][g]:.2f}" for n, g in t["foods"].items())
              + f" | distinct food values {len(fv)}, non-food geoms a ray can hit "
              f"{len([g for g in t['seen'] if g not in t['foods'].values()])}")
        for v in fv:
            food_vals[v].add(t["seed"])
    clash = 0
    for t in tabs:
        mine = {t["tex"][g] for g in t["foods"].values()}
        other = {t["tex"][g]: t["names"][g] for g in t["seen"]
                 if g not in t["foods"].values()}
        hit = {v: other[v] for v in food_vals if v not in mine and v in other}
        clash += bool(hit)
    sets = {tuple(sorted(t["tex"][g] for g in t["foods"].values())) for t in tabs}
    apple = sorted({t["tex"][g] for t in tabs for n, g in t["foods"].items()
                    if n == "apple"})
    print(f"worlds {len(tabs)}; distinct food value-sets {len(sets)}; "
          f"values that are food somewhere {len(food_vals)} of "
          f"{len({v for t in tabs for v in t['tex'].values()})} values in use; "
          f"worlds where another world's food value is a NON-food geom "
          f"{clash}/{len(tabs)}; apple reads {apple}")


def main_kind() -> None:
    """`--paint kind` (ORDERS Next-up 2, after J24 (0)): the same 23 worlds
    under `paint="kind"` — which channel does each floor food carry, and does
    any other geom a ray can hit carry it?

    RESULT (2026-11-26, box, jack-std, mujoco 3.17.0): 23 worlds
    (941-943, 281-300): every floor food (obj0-2) is on the one food channel
    and no other geom a ray can hit is on it, 23/23; one food look-set where
    the geom paint had 4. Per world: food 3, water 1, wall 4, panel 1,
    obstacle 23-27. (This reads the paint's own table; that the ROW carries
    it and nothing else is tests/test_jack_env_paint.py (b)-(c).)"""
    from experiments import fast_sim as FS
    from experiments.pilot_b2 import PILOT_RULE, make_world
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    from experiments import pilot_b2 as P
    good = 0
    sets = set()
    for seed in SEEDS:
        e = JackEnv(seed=seed, difficulty=P.DIFFICULTY, reward=P.REWARD,
                    random_needs=NEEDS_S_TRP, random_layout=True,
                    lazy_contact_audio=True, spawn_clear=SPAWN_CLEAR_R,
                    food_floor_only=PILOT_RULE["food_floor_only"],
                    eye_frame=PILOT_RULE["eye_frame"],
                    motor_frame=PILOT_RULE["motor_frame"],
                    native_substep=PILOT_RULE["native_substep"],
                    eye="row128", paint="kind")
        e.reset(seed=seed)
        w, m = e.w, e.w.model
        foods = {n: int(g) for n, (g, _) in w.drives._food.items()
                 if n in e.floor_foods}
        seen = [g for g in range(m.ngeom) if m.geom_group[g] != 3]
        fk = {n: FS.KINDS[int(w._kind_of[g])] for n, g in foods.items()}
        others = [g for g in seen if g not in foods.values()
                  and FS.KINDS[int(w._kind_of[g])] == "food"]
        count = collections.Counter(FS.KINDS[int(w._kind_of[g])] for g in seen)
        ok = set(fk.values()) == {"food"} and not others
        good += ok
        sets.add(tuple(sorted(set(fk.values()))))
        print(f"s{seed} ngeom {m.ngeom} foods "
              + " ".join(f"{n}={k}" for n, k in fk.items())
              + f" | non-food geoms on the food channel {len(others)} | "
              + " ".join(f"{k} {count[k]}" for k in FS.KINDS))
    print(f"worlds {len(SEEDS)}; every floor food on the one food channel and "
          f"nothing else on it: {good}/{len(SEEDS)}; distinct food look-sets "
          f"{len(sets)}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--paint", "kind"]:
        main_kind()
    else:
        main()
