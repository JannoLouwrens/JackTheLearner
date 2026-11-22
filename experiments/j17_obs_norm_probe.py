"""J17 suspect probe (descriptive, no learner, no ledger): what VecNormalize
hands B's critic from each eye. Random actions in PILOT_RULE's world under the
order's frame pins; SB3 VecNormalize(norm_obs, clip 10) running stats, exactly
as pilot_b.run_b builds it. Vision dims = dims that the blind arm holds at 0
on the same seed (BlindEye keeps width, zeroes content). Reports, over the
last half of the steps (stats warm): share of vision / other entries with
|z| >= 5 and at the clip (10), and the mean per-step L2 of the vision block
vs the rest. Usage: python -m experiments.j17_obs_norm_probe SEED STEPS"""
import sys, json
import numpy as np


def probe(arm, seed, steps):
    from experiments.pilot_b2 import pilot_world
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    world = pilot_world(arm, food_floor_only=True, eye_frame="body",
                        motor_frame="body", native_substep=True)
    venv = VecNormalize(DummyVecEnv([lambda: world(seed)]), norm_obs=True,
                        norm_reward=False, clip_obs=10.0)
    venv.seed(seed)
    rng = np.random.default_rng(seed)
    sp = venv.action_space
    raw, z = [], []
    venv.reset()
    for t in range(steps):
        a = rng.uniform(sp.low, sp.high, size=(1,) + sp.shape).astype(np.float32)
        o, _, _, _ = venv.step(a)
        z.append(o[0].copy()); raw.append(venv.get_original_obs()[0].copy())
    return np.array(raw), np.array(z)


def main(seed=923, steps=4000):
    out = {}
    raw_b, _ = probe("K-row128-blind", seed, steps)
    vis128 = np.all(raw_b == 0, axis=0)
    data = {a: probe(a, seed, steps) for a in ("K-row128", "K-rays16")}
    data["K-row128-blind"] = (raw_b, probe("K-row128-blind", seed, steps)[1])
    for arm, (raw, z) in data.items():
        if arm == "K-rays16":
            # same seed + same action stream -> same body: a rays16 column is
            # NON-vision iff it equals some row128 non-vision column exactly
            r128 = data["K-row128"][0][:, ~vis128]
            vis = np.array([not any(np.array_equal(raw[:, j], r128[:, k])
                                    for k in range(r128.shape[1]))
                            for j in range(raw.shape[1])])
        elif arm == "K-row128":
            vis = vis128 & ~np.all(raw == 0, axis=0)
        else:
            vis = vis128 & ~np.all(data["K-row128"][0] == 0, axis=0)
        h = z[len(z) // 2:]
        zv, zo = np.abs(h[:, vis]), np.abs(h[:, ~vis])
        r = {"dims": int(raw.shape[1]), "vision_dims": int(vis.sum()),
             "frac_vision_raw_nonzero": round(float((raw[len(raw)//2:, vis] != 0).mean()), 4) if vis.any() else None,
             "vision_|z|>=5": round(float((zv >= 5).mean()), 4) if vis.any() else None,
             "vision_at_clip": round(float((zv >= 9.999).mean()), 4) if vis.any() else None,
             "other_|z|>=5": round(float((zo >= 5).mean()), 4),
             "L2_vision_mean": round(float(np.linalg.norm(h[:, vis], axis=1).mean()), 2) if vis.any() else 0.0,
             "L2_vision_p99": round(float(np.percentile(np.linalg.norm(h[:, vis], axis=1), 99)), 2) if vis.any() else 0.0,
             "L2_other_mean": round(float(np.linalg.norm(h[:, ~vis], axis=1).mean()), 2)}
        out[arm] = r
    print(json.dumps({"seed": seed, "steps": steps, "arms": out}, indent=1))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
