"""Box test: `brain_b.save_life` on a gymnasium MuJoCo env (INBOX 11-21 20:40).

gymnasium's MujocoEnv pickles through EzPickle = a FRESH env, so save_life's
pickled env lost MjData + the env RNG (found by J10 (a), `d39687f0`, carried
there by hand in humanoid_sac_row._save_physics). save_life now carries them
itself (`carry_physics`). This tests B's own path on Hopper-v4 (mujoco 3.17.0,
jack-std = the kernel pins), plain sbx SAC, seed 0.

T = 1100 (a cut mid-episode is likely; the run prints the episode step at T).
  U1, U2   unbroken 0 -> 2T (U2 = determinism control)
  L        0 -> T, save_life, NEW process, resume_life, T more   must be CLEAN
  P_phys   the same save with carry_physics=False (= save_life before this fix)
           MUST differ
PASS = U1==U2 AND L==U1 on every item AND P_phys differs.
Prediction (written before the run, 11-21 20:50): PASS; P_phys differs from row T.

    /data/venvs/jack-std/bin/python -m experiments.b_life_mujoco_test WORK [U1|U2|L0|L1|P_phys|compare]
"""
import json
import os
import sys

from experiments.b_life_resume_test import _final, _state, _diff

SEED = 0
T = 1100
ENV = "Hopper-v4"


def _fresh_env():
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    return Monitor(gym.make(ENV))


def _model():
    from sbx import SAC
    return SAC("MlpPolicy", _fresh_env(), seed=SEED, verbose=0)


def run(work, what):
    from experiments.brain_b import save_life, resume_life
    if what in ("U1", "U2"):
        m = _model()
        m.learn(total_timesteps=2 * T)
        _final(m, os.path.join(work, what))
    elif what == "L0":
        m = _model()
        m.learn(total_timesteps=T)
        save_life(m, os.path.join(work, "L0"))
        save_life(m, os.path.join(work, "P_phys_0"), carry_physics=False)
        el = getattr(m.env.envs[0].env, "_elapsed_steps", None)
        return {"arm": what, "num_timesteps": int(m.num_timesteps), "elapsed_in_episode": el}
    elif what in ("L1", "P_phys"):
        src = "L0" if what == "L1" else "P_phys_0"
        m = resume_life(os.path.join(work, src), _fresh_env())
        m.learn(total_timesteps=T, reset_num_timesteps=False)
        _final(m, os.path.join(work, what))
    else:
        raise SystemExit(f"unknown arm {what}")
    return {"arm": what, "num_timesteps": int(m.num_timesteps),
            "n_updates": int(getattr(m, "_n_updates", -1))}


def _load(path):
    from sbx import SAC
    m = SAC.load(os.path.join(path, "model.zip"), env=_fresh_env())
    m.load_replay_buffer(os.path.join(path, "replay.pkl"))
    return m


def compare(work):
    import experiments.b_row1b_resume_twin as twin
    twin._model = _load                # the twin's reader, on this env (it hard-codes Pendulum)
    arms = ("U1", "U2", "L1", "P_phys")
    S = {a: _state(os.path.join(work, a)) for a in arms}
    out = {a: _diff(S["U1"][0], S[a][0]) for a in arms[1:]}
    det = not out["U2"]["differing"]
    clean = not out["L1"]["differing"]
    caught = bool(out["P_phys"]["differing"])
    out["reading"] = {"control_U1_eq_U2": det, "L_clean": clean, "P_phys_differs": caught,
                      "P_phys_differing": out["P_phys"]["differing"],
                      "verdict": ("VOID: box not deterministic" if not det else
                                  "PASS" if clean and caught else "FAIL")}
    return out


if __name__ == "__main__":
    work, what = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    r = compare(work) if what == "compare" else run(work, what)
    with open(os.path.join(work, f"{what}.json"), "w") as f:
        json.dump(r, f, indent=1, default=str)
    print("RESULT " + json.dumps(r.get("reading", r), default=str)[:3000], flush=True)
