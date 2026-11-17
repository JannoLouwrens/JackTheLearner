"""J6 (2b) piece 1: C contact scans == today's Python on live sub-steps.

On 3 seeds (153-155), shipped cal, lazy_contact_audio=True, 150 random
decisions each (40 sub-steps a decision), every call of `W0._grounded` and of
`drives._contact_partners` (both the drive layer's jack mask) is checked
against `experiments.native_substep`'s C twin on the SAME mjData at the SAME
instant. Must match on every call. Control (must FAIL to match): the C
grounded scan fed the ground mask with every ground geom cleared — it must
disagree with Python on at least one sub-step per seed where Jack is grounded.
"""
import numpy as np

from experiments import drives
from experiments.jack_env import JackEnv
from experiments.native_substep import NativeContacts

SEEDS = (153, 154, 155)
DECISIONS = 150


def _run(seed):
    env = JackEnv(seed=seed, lazy_contact_audio=True)
    env.reset(seed=seed)
    w = env.w
    nc = NativeContacts(w._body_mask, w._ground_mask)
    dead = NativeContacts(w._body_mask, np.zeros_like(w._ground_mask))
    stats = dict(g=0, g_true=0, g_bad=0, ctrl_diff=0, p=0, p_bad=0, p_nonempty=0)
    orig_g = w._grounded

    def g_wrap():
        r = orig_g()
        stats["g"] += 1
        stats["g_true"] += r
        stats["g_bad"] += (nc.grounded(w.data) != r)
        stats["ctrl_diff"] += (dead.grounded(w.data) != r)
        return r
    w._grounded = g_wrap

    orig_p = drives._contact_partners
    ptrs = {}

    def p_wrap(data, mask):
        r = orig_p(data, mask)
        key = id(mask)
        if key not in ptrs:
            arr = np.ascontiguousarray(mask, dtype=np.uint8)
            ptrs[key] = (arr, nc.ffi.from_buffer("uint8_t[]", arr))
        row = nc.partners_row(data, ptrs[key][1])
        stats["p"] += 1
        stats["p_nonempty"] += bool(r)
        stats["p_bad"] += (set(np.flatnonzero(row).tolist()) != r)
        return r
    drives._contact_partners = p_wrap
    try:
        rng = np.random.default_rng(seed)
        for _ in range(DECISIONS):
            _, _, term, trunc, _ = env.step(
                rng.uniform(-1, 1, env.action_space.shape))
            if term or trunc:
                env.reset()
                w = env.w
    finally:
        drives._contact_partners = orig_p
    return stats


def test_native_contacts_identical():
    for s in SEEDS:
        st = _run(s)
        print(s, st)
        assert st["g"] > 0 and st["p"] > 0
        assert st["g_bad"] == 0 and st["p_bad"] == 0
        assert st["g_true"] > 0 and st["p_nonempty"] > 0
        assert st["ctrl_diff"] > 0          # the control must fail


if __name__ == "__main__":
    test_native_contacts_identical()
    print("PASS")
