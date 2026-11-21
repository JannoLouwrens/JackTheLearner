"""J10 (a) — is the row's resume clean? (ruling 72; box-minutes, no new seed)

The SAME short run three times through `humanoid_sac_row.train_segment` (the
row's own resume path, unedited):
  U1  unbroken, 0 -> 2T in one segment
  U2  unbroken again (determinism control: if U1 != U2 the box itself is not
      bit-reproducible and a twin difference says nothing)
  R   segment 0 to T, save, load, segment 1 to 2T (the row's resume)
Compared at 2T, item by item: actor / critic / target / ent_coef params, the
three optimiser states, replay contents IN ORDER (obs, actions, rewards,
dones, next_obs), buffer pos, num_timesteps, n_updates, sbx's PRNG key, and
the first-differing replay row (where the resumed run's experience departs).
Toy size: learning_starts = T/2 so both halves train; buffer 4T. The rule's
SAC_KW otherwise. Ours beside the row: the row's segments are 500k; here T is
a few thousand — this tests the path, not the scale.

    python -m experiments.j10_resume_twin WORK T [U1|U2|R|RC|compare]

RC (J10 (a) fix): R through train_segment(carry_life=True); R stays the
planted broken resume and must still differ.
"""
import json
import os
import sys


def _kw(T):
    from experiments.humanoid_sac_row import SAC_KW
    return dict(SAC_KW, learning_starts=T // 2, buffer_size=4 * T)


def run_arm(work, T, arm):
    from experiments.humanoid_sac_row import train_segment
    kw = _kw(T)
    if arm in ("U1", "U2"):
        return train_segment(0, os.path.join(work, arm), None, 0, seg_steps=2 * T,
                             total=2 * T, eval_at=(), sac_kw=kw, wall_h=1.0)
    c = arm == "RC"                    # RC: the same resume with carry_life=True (the fix)
    a, b = os.path.join(work, arm + "0"), os.path.join(work, arm + "1")
    r0 = train_segment(0, a, None, 0, seg_steps=T, total=2 * T, eval_at=(), sac_kw=kw, wall_h=1.0,
                       carry_life=c)
    r1 = train_segment(0, b, a, 1, seg_steps=T, total=2 * T, eval_at=(), sac_kw=kw, wall_h=1.0,
                       carry_life=c)
    return {"seg0": r0, "seg1": r1}


def _state(d):
    import pickle
    import numpy as np
    import jax
    from sbx import SAC
    from experiments.humanoid_sac_row import make_env, _ckpt_paths
    mp, rp = _ckpt_paths(d)
    m = SAC.load(mp, env=make_env())
    p = m.policy
    leaves = lambda t: np.concatenate([np.asarray(x, dtype=np.float64).ravel()
                                       for x in jax.tree_util.tree_leaves(t)] or [np.zeros(0)])
    with open(rp, "rb") as f:
        buf = pickle.load(f)
    n = buf.buffer_size if buf.full else buf.pos
    out = {
        "actor_params": leaves(p.actor_state.params),
        "critic_params": leaves(p.qf_state.params),
        "target_params": leaves(p.qf_state.target_params),
        "ent_coef_params": leaves(m.ent_coef_state.params),
        "actor_opt": leaves(p.actor_state.opt_state),
        "critic_opt": leaves(p.qf_state.opt_state),
        "ent_coef_opt": leaves(m.ent_coef_state.opt_state),
        "prng_key": np.asarray(jax.random.key_data(m.key) if hasattr(jax.random, "key_data")
                               and jax.dtypes.issubdtype(m.key.dtype, jax.dtypes.prng_key)
                               else m.key).ravel().astype(np.float64),
    }
    for k in ("observations", "actions", "rewards", "dones", "next_observations", "timeouts"):
        if hasattr(buf, k):
            out["replay_" + k] = np.asarray(getattr(buf, k))[:n]
    meta = {"num_timesteps": int(m.num_timesteps), "n_updates": int(getattr(m, "_n_updates", -1)),
            "buffer_pos": int(buf.pos), "buffer_full": bool(buf.full)}
    return out, meta


def compare(work):
    import numpy as np
    arms = ["U1", "U2", "R1"] + (["RC1"] if os.path.isdir(os.path.join(work, "RC1")) else [])
    S = {k: _state(os.path.join(work, k)) for k in arms}

    def diff(x, y):
        sx, mx = S[x]; sy, my = S[y]
        items = {}
        for k in sx:
            a, b = sx[k], sy[k]
            if a.shape != b.shape:
                items[k] = {"equal": False, "why": f"shape {a.shape} vs {b.shape}"}
                continue
            eq = bool(np.array_equal(a, b))
            it = {"equal": eq}
            if not eq:
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                it["max_abs"] = float(d.max())
                if k.startswith("replay_"):
                    rows = np.where(d.reshape(d.shape[0], -1).max(axis=1) > 0)[0]
                    it["first_row"] = int(rows[0]); it["rows_differing"] = int(rows.size)
            items[k] = it
        return {"items": items, "meta": {x: mx, y: my},
                "differing": [k for k, v in items.items() if not v["equal"]]}

    ctrl = diff("U1", "U2")
    twin = diff("U1", "R1")
    deterministic = not ctrl["differing"]
    reading = ("CLEAN: the resumed run is bit-identical to the unbroken one" if deterministic
               and not twin["differing"] else
               "BOX NOT DETERMINISTIC: the control differs, the twin cannot be read" if not deterministic
               else "RESUME DIFFERS: " + ", ".join(twin["differing"]))
    out = {"control_U1_vs_U2": ctrl, "twin_U1_vs_R": twin, "reading": reading}
    if "RC1" in S:                     # the fix: must read CLEAN where R (the planted break) differs
        fix = diff("U1", "RC1")
        out["twin_U1_vs_RC"] = fix
        out["reading_RC"] = ("CLEAN" if deterministic and not fix["differing"] else
                             "CONTROL DIFFERS" if not deterministic else
                             "RC DIFFERS: " + ", ".join(fix["differing"]))
    return out


if __name__ == "__main__":
    work, T, what = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(work, exist_ok=True)
    r = compare(work) if what == "compare" else run_arm(work, T, what)
    with open(os.path.join(work, f"{what}.json"), "w") as f:
        json.dump(r, f, indent=1, default=str)
    print("RESULT " + json.dumps(r, default=str)[:4000], flush=True)
