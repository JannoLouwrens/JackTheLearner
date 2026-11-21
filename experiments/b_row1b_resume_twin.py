"""Next-up 4 (PUBLISHED_PRIORS §20) — is brain B's stop-resume clean on row 1b?

Box-minutes, no new seed, no ledger. Row 1b's `ours` arm EXACTLY as the row
builds it (`brain_b.make_row1("ours", seed)`: SBX SAC + layer_norm=True on
Pendulum-v1, lr 1e-3, SB3 defaults otherwise — learning_starts 100,
train_freq 1, gradient_steps 1). T decisions, then T more = T more updates:
  U1  unbroken 0 -> 2T in one process
  U2  unbroken again (determinism control: U1 != U2 => the box is not
      bit-reproducible and the twin says nothing)
  R   0 -> T, SB3 save() + save_replay_buffer(), NEW process, SAC.load +
      load_replay_buffer, learn(T, reset_num_timesteps=False) — SB3's own
      resume path, what a moved Jack would do today
  RE  as R, but the save also carries the env's RNG state, the pendulum's
      physical state and `_last_obs` (the life in flight), restored on load
      — the candidate fix, so a RESUME DIFFERS reading names what must move
  RE2 as RE, plus numpy's GLOBAL RNG state (added after RE read 13:45: RE
      departs one row later, at the first update -> SB3's ReplayBuffer.sample
      draws batch indices from np.random, which no save carries)
  RE3 as RE2, plus sbx's TRAINING key `model.key` (added 11-21 14:27: a probe
      showed SAC.load leaves model.key = [0, 0] while the live key at T is not;
      policy.key/noise_key DO load). Every arm now also writes key.npy beside
      model.zip and the compare reads the key from THAT: before, `_state` read
      the key after SAC.load, so "prng_key equal in every arm" compared [0,0]
      with [0,0] — a check that could not fail.
Compared at 2T item by item (actor/critic/target/ent_coef params, the three
optimiser states, sbx PRNG key, replay contents IN ORDER, first differing row).
PASS for an arm = bit-identical to U1 with the control clean. Prediction
written before running (11-21 13:40; RE3 added 14:27 before its run: RE3 reads CLEAN): R differs from replay row T on (SB3
drops `_last_obs`; a fresh env RNG), RE reads CLEAN.
T = 1000 is a multiple of Pendulum's 200-step cap, so no episode is cut at the
save — the only thing in flight is the env's RNG and the next start state.

    /data/venvs/jack-std/bin/python -m experiments.b_row1b_resume_twin WORK T [U1|U2|R0|R1|RE1|RE2|RE3|compare]
"""
import json
import os
import pickle
import sys

SEED = 0


def _model(path=None):
    from experiments.brain_b import make_row1
    if path is None:
        return make_row1("ours", SEED)
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    from sbx import SAC
    m = SAC.load(os.path.join(path, "model.zip"), env=Monitor(gym.make("Pendulum-v1")))
    m.load_replay_buffer(os.path.join(path, "replay.pkl"))
    return m


def _key_data(key):
    import numpy as np
    import jax
    if hasattr(jax.random, "key_data") and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    return np.asarray(key)


def _save(m, d, life=False):
    os.makedirs(d, exist_ok=True)
    m.save(os.path.join(d, "model.zip"))
    m.save_replay_buffer(os.path.join(d, "replay.pkl"))
    __import__("numpy").save(os.path.join(d, "key.npy"), _key_data(m.key))
    if life:
        u = m.env.envs[0].unwrapped
        with open(os.path.join(d, "life.pkl"), "wb") as f:
            pickle.dump({"rng": u.np_random.bit_generator.state, "state": u.state.copy(),
                         "last_obs": m._last_obs.copy(),
                         "episode_starts": m._last_episode_starts.copy(),
                         "np_global": __import__("numpy").random.get_state()}, f)


def _restore_life(m, d, global_np=False, carry_key=False):
    with open(os.path.join(d, "life.pkl"), "rb") as f:
        L = pickle.load(f)
    m.env.reset()                       # satisfy the order check; then overwrite the life
    u = m.env.envs[0].unwrapped
    u.np_random.bit_generator.state = L["rng"]
    u.state = L["state"]
    m._last_obs = L["last_obs"]
    m._last_episode_starts = L["episode_starts"]
    if global_np:                       # RE2: SB3's replay sampler draws from numpy's GLOBAL RNG
        __import__("numpy").random.set_state(L["np_global"])
    if carry_key:                       # RE3: sbx's training key, which SAC.load leaves at [0, 0]
        import jax.numpy as jnp
        m.key = jnp.asarray(__import__("numpy").load(os.path.join(d, "key.npy")))


def run(work, T, what):
    if what in ("U1", "U2"):
        m = _model()
        m.learn(total_timesteps=2 * T)
        _save(m, os.path.join(work, what))
    elif what == "R0":
        m = _model()
        m.learn(total_timesteps=T)
        _save(m, os.path.join(work, "R0"), life=True)
    elif what in ("R1", "RE1", "RE2", "RE3"):
        m = _model(os.path.join(work, "R0"))
        if what != "R1":
            _restore_life(m, os.path.join(work, "R0"), global_np=(what in ("RE2", "RE3")),
                          carry_key=(what == "RE3"))
        m.learn(total_timesteps=T, reset_num_timesteps=False)
        _save(m, os.path.join(work, what))
    return {"arm": what, "num_timesteps": int(m.num_timesteps),
            "n_updates": int(getattr(m, "_n_updates", -1))}


def _state(d):
    import numpy as np
    import jax
    m = _model(d)
    p = m.policy
    leaves = lambda t: np.concatenate([np.asarray(x, dtype=np.float64).ravel()
                                       for x in jax.tree_util.tree_leaves(t)] or [np.zeros(0)])
    key = np.load(os.path.join(d, "key.npy"))  # the LIVE key at save, not SAC.load's [0, 0]
    buf = m.replay_buffer
    n = buf.buffer_size if buf.full else buf.pos
    out = {"actor_params": leaves(p.actor_state.params),
           "critic_params": leaves(p.qf_state.params),
           "target_params": leaves(p.qf_state.target_params),
           "ent_coef_params": leaves(m.ent_coef_state.params),
           "actor_opt": leaves(p.actor_state.opt_state),
           "critic_opt": leaves(p.qf_state.opt_state),
           "ent_coef_opt": leaves(m.ent_coef_state.opt_state),
           "prng_key": np.asarray(key).ravel().astype(np.float64)}
    for k in ("observations", "actions", "rewards", "dones", "next_observations", "timeouts"):
        out["replay_" + k] = np.asarray(getattr(buf, k))[:n]
    return out, {"num_timesteps": int(m.num_timesteps), "n_updates": int(getattr(m, "_n_updates", -1)),
                 "buffer_pos": int(buf.pos)}


def compare(work):
    import numpy as np
    S = {k: _state(os.path.join(work, k)) for k in ("U1", "U2", "R1", "RE1", "RE2", "RE3")}

    def diff(x, y):
        (sx, mx), (sy, my) = S[x], S[y]
        items = {}
        for k in sx:
            a, b = sx[k], sy[k]
            if a.shape != b.shape:
                items[k] = {"equal": False, "why": f"shape {a.shape} vs {b.shape}"}
                continue
            it = {"equal": bool(np.array_equal(a, b))}
            if not it["equal"]:
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                it["max_abs"] = float(d.max())
                if k.startswith("replay_"):
                    rows = np.where(d.reshape(d.shape[0], -1).max(axis=1) > 0)[0]
                    it["first_row"], it["rows_differing"] = int(rows[0]), int(rows.size)
            items[k] = it
        return {"items": items, "meta": {x: mx, y: my},
                "differing": [k for k, v in items.items() if not v["equal"]]}

    ctrl = diff("U1", "U2")
    det = not ctrl["differing"]

    def read(t):
        if not det:
            return "BOX NOT DETERMINISTIC: control differs, twin unreadable"
        return "CLEAN" if not t["differing"] else "RESUME DIFFERS: " + ", ".join(t["differing"])
    r, re_, re2, re3 = diff("U1", "R1"), diff("U1", "RE1"), diff("U1", "RE2"), diff("U1", "RE3")
    return {"control_U1_vs_U2": ctrl, "twin_U1_vs_R": r, "twin_U1_vs_RE": re_, "twin_U1_vs_RE2": re2,
            "twin_U1_vs_RE3": re3, "reading_R": read(r), "reading_RE": read(re_),
            "reading_RE2": read(re2), "reading_RE3": read(re3)}


if __name__ == "__main__":
    work, T, what = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(work, exist_ok=True)
    r = compare(work) if what == "compare" else run(work, T, what)
    with open(os.path.join(work, f"{what}.json"), "w") as f:
        json.dump(r, f, indent=1, default=str)
    print("RESULT " + json.dumps(r, default=str)[:3000], flush=True)
