"""Box test of B's own checkpoint path: `brain_b.save_life` / `resume_life`.

Next-up 4 follow-on (ORDERS, 11-21). The twin (`b_row1b_resume_twin.py`) found
the items SB3/sbx's save drops and patched them inside the twin script; this
puts them in brain_b.py and tests THAT code. Box-minutes, no ledger.

Row 1b's `ours` arm exactly (`make_row1("ours", 0)`), Pendulum-v1.
T = 1100 — NOT a multiple of the 200-step cap, so the save cuts an episode at
step 100 of 200 (the twin saved only on a boundary: a cut life was untested;
TimeLimit's elapsed-step counter and Monitor's running return are now in play).

  U1, U2   unbroken 0 -> 2T (U2 = the box's determinism control)
  L        0 -> T, save_life, NEW process, resume_life, T more     must be CLEAN
  planted  each one item left out of the save, resumed the same way, MUST differ:
           P_key (no model.key), P_np (no numpy/python global RNG), P_env (no
           pickled env: a fresh reset life)
  R        SB3's own save()/load() with nothing carried               must differ

PASS = control U1==U2 clean AND L bit-identical to U1 on every item AND every
planted arm AND R differ. Prediction (written before the run, 11-21 18:25):
PASS; P_env differs at replay row T.

Compared at 2T: actor/critic/target/ent_coef params, the three optimiser
states, the live training key (key.npy written by save_life), replay contents
in order, plus Monitor's episode returns (the cut episode must finish with the
same return in L as in U1).

    /data/venvs/jack-std/bin/python -m experiments.b_life_resume_test WORK [U1|U2|L0|L1|P_key|P_np|P_env|R1|compare]
"""
import json
import os
import sys

SEED = 0
T = 1100
PLANTS = {"P_key": dict(carry_key=False), "P_np": dict(carry_np=False),
          "P_env": dict(carry_env=False)}


def _fresh_env():
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    return Monitor(gym.make("Pendulum-v1"))


def _final(m, d):
    from experiments.brain_b import save_life
    save_life(m, d)
    ep = [float(x) for x in m.env.envs[0].get_episode_rewards()]
    with open(os.path.join(d, "episodes.json"), "w") as f:
        json.dump(ep, f)


def run(work, what):
    from experiments.brain_b import make_row1, save_life, resume_life
    if what in ("U1", "U2"):
        m = make_row1("ours", SEED)
        m.learn(total_timesteps=2 * T)
        _final(m, os.path.join(work, what))
    elif what == "L0":
        m = make_row1("ours", SEED)
        m.learn(total_timesteps=T)
        save_life(m, os.path.join(work, "L0"))
        for name, kw in PLANTS.items():      # same model, same instant, one item left out
            save_life(m, os.path.join(work, name + "_0"), **kw)
        os.makedirs(os.path.join(work, "R0"), exist_ok=True)
        m.save(os.path.join(work, "R0", "model.zip"))   # SB3's own path, nothing carried
        m.save_replay_buffer(os.path.join(work, "R0", "replay.pkl"))
        elapsed = m.env.envs[0].env._elapsed_steps if hasattr(m.env.envs[0].env, "_elapsed_steps") else None
        return {"arm": what, "num_timesteps": int(m.num_timesteps), "elapsed_in_episode": elapsed}
    elif what in ("L1",) + tuple(PLANTS):
        src = "L0" if what == "L1" else what + "_0"
        m = resume_life(os.path.join(work, src), _fresh_env())
        m.learn(total_timesteps=T, reset_num_timesteps=False)
        _final(m, os.path.join(work, what))
    elif what == "R1":
        from sbx import SAC
        m = SAC.load(os.path.join(work, "R0", "model.zip"), env=_fresh_env())
        m.load_replay_buffer(os.path.join(work, "R0", "replay.pkl"))
        m.learn(total_timesteps=T, reset_num_timesteps=False)
        _final(m, os.path.join(work, what))
    else:
        raise SystemExit(f"unknown arm {what}")
    return {"arm": what, "num_timesteps": int(m.num_timesteps),
            "n_updates": int(getattr(m, "_n_updates", -1))}


def _state(d):
    import numpy as np
    from experiments.b_row1b_resume_twin import _state as twin_state
    s, meta = twin_state(d)          # reads key.npy (the live key), replay in order
    with open(os.path.join(d, "episodes.json")) as f:
        s["monitor_episode_returns"] = np.asarray(json.load(f), dtype=np.float64)
    return s, meta


def _diff(sx, sy):
    import numpy as np
    items = {}
    for k in sx:
        a, b = sx[k], sy[k]
        if a.shape != b.shape:
            items[k] = {"equal": False, "why": f"shape {a.shape} vs {b.shape}"}
            continue
        it = {"equal": bool(np.array_equal(a, b))}
        if not it["equal"]:
            dd = np.abs(a.astype(np.float64) - b.astype(np.float64))
            it["max_abs"] = float(dd.max())
            if k.startswith("replay_"):
                rows = np.where(dd.reshape(dd.shape[0], -1).max(axis=1) > 0)[0]
                it["first_row"], it["rows_differing"] = int(rows[0]), int(rows.size)
        items[k] = it
    return {"differing": [k for k, v in items.items() if not v["equal"]], "items": items}


def compare(work):
    arms = ("U1", "U2", "L1", "R1") + tuple(PLANTS)
    S = {a: _state(os.path.join(work, a)) for a in arms}
    out = {a: _diff(S["U1"][0], S[a][0]) for a in arms[1:]}
    out["meta"] = {a: S[a][1] for a in arms}
    det = not out["U2"]["differing"]
    clean = not out["L1"]["differing"]
    caught = {a: bool(out[a]["differing"]) for a in ("R1",) + tuple(PLANTS)}
    out["reading"] = {"control_U1_eq_U2": det, "L_clean": clean, "must_differ": caught,
                      "verdict": ("VOID: box not deterministic" if not det else
                                  "PASS" if clean and all(caught.values()) else "FAIL")}
    return out


if __name__ == "__main__":
    work, what = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    r = compare(work) if what == "compare" else run(work, what)
    with open(os.path.join(work, f"{what}.json"), "w") as f:
        json.dump(r, f, indent=1, default=str)
    print("RESULT " + json.dumps(r.get("reading", r), default=str)[:3000], flush=True)
