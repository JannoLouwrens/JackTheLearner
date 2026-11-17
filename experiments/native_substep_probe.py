import sys, time, numpy as np
sys.path.insert(0, "/home/opc/jackthelearner")
from experiments.jack_env import JackEnv
from experiments.native_substep import apply_native_substep
for seed in (153, 154, 155):
    a, b = JackEnv(seed=seed, lazy_contact_audio=True), JackEnv(seed=seed, lazy_contact_audio=True)
    oa, ob = a.reset(), b.reset()
    apply_native_substep(b.w)
    rng = np.random.default_rng(seed)
    bad = deaths = 0; ta = tb = 0.0
    for t in range(300):
        act = rng.uniform(-1, 1, a.action_space.shape if hasattr(a, "action_space") else 8)
        t0 = time.process_time(); ra = a.step(act); ta += time.process_time() - t0
        t0 = time.process_time(); rb = b.step(act); tb += time.process_time() - t0
        deaths += int(a.w.died_this_decision)
        same = all(np.array_equal(np.asarray(x, dtype=object) if not isinstance(x, np.ndarray) else x,
                                  np.asarray(y, dtype=object) if not isinstance(y, np.ndarray) else y)
                   for x, y in zip(ra[:3], rb[:3])) and np.array_equal(a.w.data.qpos, b.w.data.qpos)
        bad += int(not same)
    print(f"seed {seed}: differing decisions {bad}/300, deaths {deaths}, "
          f"shipped {300/ta:.1f} native {300/tb:.1f} dec/CPU-s", flush=True)
