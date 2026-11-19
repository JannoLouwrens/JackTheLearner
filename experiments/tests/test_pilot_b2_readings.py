"""J11 item 4 (11-19 05:20, before any pilot run is home): THE RULE's readings
(`pilot_b2.pilot_readings`, PILOT_B2.md §3 frozen `e2c3c378`) on planted books
with known answers.

Base book (40 runs, seeds 402-406): floors starve; B0 SLIDES (fed early, unfed
noop-length lives late, 5/5); K SUSTAINED (2 capped-fed lives in a row in 3 of
5 runs); P NEAR RANDOM (one meal every life, no trend); G EATS BUT STARVES (3
lives in a row of >= 5 meals, never capped, 3 of 5); KP INCOMPLETE (2 runs cut
by our timeout, rc -9, at 100k).

Must-fail controls, each one clause: K at 2 of 5 -> not SUSTAINED; a noop run
crossing W=1 -> VOID broken; held8 crossing at W=2 once -> B0/P/G VOID floor
moved while K/KP stand; K-lr0 once -> K/KP floor moved; K-lr0 twice -> broken;
two B0 crashes rc 1 -> broken, but two rc -9 cuts are not; held8 crossing θ in
2 of 5 -> G VOID θ too easy; a missing run -> no reading at all."""
import copy

from experiments import pilot_b2 as P

SEEDS = (402, 403, 404, 405, 406)


def life(dec, cause, meals, intake, drain, sustained):
    return {"decisions": dec, "cause": cause, "floor_meals": meals, "apples": 0,
            "intake": intake, "basal_drain": drain, "sustained": sustained,
            "seconds": dec / 4.0, "reward": float(meals)}


STARVE = life(4000, "hunger", 0, 0.0, 1.0, False)
FED = life(9000, "capped", 12, 2.0, 1.0, True)
EARLY = life(2000, "hunger", 1, 0.2, 1.0, False)
ONE = life(3000, "hunger", 1, 0.2, 1.0, False)
EAT6 = life(4000, "hunger", 6, 0.8, 1.0, False)


def run(lives, T=P.PILOT_T, **kw):
    rest = T - sum(l["decisions"] for l in lives)
    if rest > 0:
        lives = lives + [life(rest, "unfinished", 0, 0.0, 1.0, False)]
    return {"lives": lives, "T": T, "rc": 0, **kw}


def base():
    runs = {}
    for i, s in enumerate(SEEDS):
        r = lambda a, v: runs.__setitem__(f"{a}_s{s}", v)  # noqa: E731
        r("noop", run([STARVE] * 70))
        r("held8", run([STARVE] * 70))
        r("K-lr0", run([STARVE] * 70))
        r("B0", run([EARLY] * 5 + [STARVE] * 60))
        r("K", run([STARVE] * 6 + ([FED, FED] if i < 3 else [STARVE, STARVE]) + [STARVE] * 50))
        r("P", run([ONE] * 90))
        r("G", run([STARVE] * 6 + ([EAT6] * 3 if i < 3 else [STARVE] * 3) + [STARVE] * 50))
        if i < 2:
            r("KP", {"lives": [STARVE] * 25, "T": P.PILOT_T, "rc": -9, "cut": True,
                     "decisions_reached": 100_000, "error": "rc=-9: timeout"})
        else:
            r("KP", run([STARVE] * 70))
    return runs


def read(runs):
    return P.pilot_readings(runs, SEEDS)


def test_base_book():
    res = read(base())
    assert res["verdict"] == "READ", res
    assert res["arms"] == {"B0": "SLIDES", "K": "SUSTAINED", "P": "NEAR RANDOM",
                           "G": "EATS BUT STARVES", "KP": "INCOMPLETE"}, res["arms"]
    assert "SUSTAINED" in P.readings_md(res)


def test_two_of_five_is_not_sustained():
    runs = base()
    runs["K_s404"] = run([STARVE] * 75)
    assert read(runs)["arms"]["K"] != "SUSTAINED"


def test_noop_crossing_is_broken():
    runs = base()
    runs["noop_s402"] = run([STARVE] * 6 + [FED] + [STARVE] * 50)
    assert read(runs)["verdict"] == "VOID — broken"


def test_held8_moved_voids_its_arms_only():
    runs = base()
    runs["held8_s403"] = run([STARVE] * 6 + [FED, FED] + [STARVE] * 50)
    a = read(runs)["arms"]
    assert a["B0"] == a["P"] == a["G"] == "VOID — floor moved"
    assert a["K"] == "SUSTAINED"


def test_klr0_once_moves_k_floor_twice_is_broken():
    runs = base()
    runs["K-lr0_s405"] = run([STARVE] * 6 + [FED, FED] + [STARVE] * 50)
    a = read(runs)["arms"]
    assert a["K"] == a["KP"] == "VOID — floor moved" and a["B0"] == "SLIDES"
    runs["K-lr0_s406"] = runs["K-lr0_s405"]
    assert read(runs)["verdict"] == "VOID — broken"


def test_crash_vs_cut():
    runs = base()
    for s in (402, 403):
        runs[f"B0_s{s}"] = dict(runs[f"B0_s{s}"], rc=1, error="rc=1: nan")
    assert read(runs)["verdict"] == "VOID — broken"
    runs = base()                      # KP's two rc -9 cuts did not break the book
    assert read(runs)["verdict"] == "READ"


def test_theta_too_easy():
    runs = base()
    for s in (402, 403):
        runs[f"held8_s{s}"] = run([STARVE] * 6 + [EAT6] * 3 + [STARVE] * 50)
    assert read(runs)["arms"]["G"].startswith("VOID — θ too easy")


def test_missing_run_refuses():
    runs = base()
    del runs["P_s406"]
    try:
        read(runs)
    except ValueError as e:
        assert "P_s406" in str(e)
    else:
        raise AssertionError("a partial book must refuse")


def test_slide_needs_noop_length_lives():
    runs = base()                       # B0's late lives now short (500 s < 0.9 x 1000)
    for s in SEEDS:
        runs[f"B0_s{s}"] = run([EARLY] * 5 + [life(2000, "hunger", 0, 0.0, 1.0, False)] * 120)
    assert read(runs)["arms"]["B0"] == "NEAR RANDOM"


def main():
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            n += 1
            print("ok", k)
    print(f"PASS {n}/{n}")


if __name__ == "__main__":
    main()
