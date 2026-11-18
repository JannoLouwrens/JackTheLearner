"""J11 / ruling 39: the pilot's runs carry the body-frame eye (and the motor
frame the rule names) from PILOT_RULE all the way into the child's JackEnv.

Before this (11-18 12:40) `pilot_b2.make_world` passed neither `eye_frame` nor
`motor_frame`, and `pilot_kernel` checked neither: a rule written to the
order's letter would have launched 40 runs on the COMPASS eye.

Checks (each through the real code path):
  1. identity: default make_world / pilot_world = world frames (the f-card's
     world is unchanged by this commit).
  2. pilot_world(..., "body", "body") builds a body/body env, for a learner
     arm and a floor arm.
  3. must-fail: pilot_kernel refuses a rule with no eye_frame, with
     eye_frame "world", and with no motor_frame (all before any submit).
  4. the job body carries the frames.
  5. end to end: one child (held8, seed 901 -- outside every registered range,
     T=300, no learner) run through run_pilot_parallel with body/body, and the
     child's world stamp -- read off the env, not the caller -- says body/body.
     Control: the same child with defaults stamps world/world.
Run with /data/venvs/jack-std/bin/python or the project venv.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments import pilot_b2 as P  # noqa: E402


def main():
    ok = True
    e = P.make_world(901)
    e2 = P.pilot_world("K", True)(901)
    g1 = (e.eye_frame, e.motor_frame, e2.eye_frame, e2.motor_frame) == ("world",) * 4
    print(f"1 identity (defaults world): {g1}")
    ok &= g1

    eb = P.pilot_world("K", True, "body", "body")(901)
    fb = P.pilot_world("held8", True, "body", "world")(901)
    g2 = ((eb.eye_frame, eb.motor_frame) == ("body", "body")
          and (fb.eye_frame, fb.motor_frame) == ("body", "world")
          and eb.food_floor_only is True and eb.move_cost_dial is True)
    print(f"2 body frames reach the env (learner + floor): {g2}")
    ok &= g2

    base = {"seeds": (402,), "food_floor_only": True}
    bad = {"no eye_frame": dict(base, motor_frame="body"),
           "eye world": dict(base, eye_frame="world", motor_frame="body"),
           "no motor_frame": dict(base, eye_frame="body"),
           "motor junk": dict(base, eye_frame="body", motor_frame="compass")}
    for k, rule in bad.items():
        try:
            P.pilot_kernel("p0", rule=rule)
            print(f"3 refusal [{k}]: RAN -> BAD")
            ok = False
        except RuntimeError as err:
            print(f"3 refusal [{k}]: ok ({str(err)[:50]}...)")

    body = P.pilot_job_body(P.pilot_groups((402,))["p0"], True, 300, "t", "body", "world")
    compile(body, "job", "exec")
    g4 = ", 'body', 'world')" in body
    print(f"4 job body carries frames: {g4}")
    ok &= g4

    with tempfile.TemporaryDirectory(dir="/data/tmp" if Path("/data/tmp").is_dir() else None) as d:
        r = P.run_pilot_parallel([("held8", 901)], d, True, T=300,
                                 eye_frame="body", motor_frame="body")
        rec = r["held8_s901"]
        st = rec.get("world", {})
        g5 = (rec["rc"] == 0 and st.get("eye_frame") == "body"
              and st.get("motor_frame") == "body" and rec.get("eye_frame") == "body")
        print(f"5 child body/body: rc {rec['rc']} stamp eye={st.get('eye_frame')} "
              f"motor={st.get('motor_frame')} -> {g5}")
        r0 = P.run_pilot_parallel([("held8", 901)], d + "/ctl", True, T=300)
        st0 = r0["held8_s901"].get("world", {})
        g6 = st0.get("eye_frame") == "world" and st0.get("motor_frame") == "world"
        print(f"5 control child defaults: stamp eye={st0.get('eye_frame')} "
              f"motor={st0.get('motor_frame')} -> {g6}")
        ok &= g5 and g6

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
