"""J20 (4): eye_streak.collect and find_himself.collect cast rays through the
eye's own compat (`fast_sim._multi_ray`), so they run on mujoco 3.2.3 (box
project venv) and 3.17.0 (jack-std, every Kaggle kernel).

  - on any version: the routed collect runs and writes its file;
  - on 3.2.3: its output equals the pre-J20 code's (the commit before this
    one, read from git) bit for bit — the cutoff change (RAY_MAX_RANGE ->
    mjMAXVAL) is masked as a miss, so nothing moved;
  - on >= 3.3 (control): the pre-J20 code FAILS on the new signature, so the
    test could have failed and the route is what makes it pass.

Run directly (no pytest on the box):
    $PY experiments/tests/test_j20_ray_callers.py [OLD_REV]
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OLD_REV = sys.argv[1] if len(sys.argv) > 1 else "a98de521"   # before J20 (4)


def _old_module(name: str, tmp: str):
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{OLD_REV}:experiments/{name}.py"],
                         check=True, capture_output=True, text=True).stdout
    p = Path(tmp) / f"j20old_{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(f"j20old_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a)


def _case(name: str, env_var: str, n: int, tmp: str) -> str:
    import mujoco
    new_sig = tuple(int(x) for x in mujoco.__version__.split(".")[:2]) >= (3, 3)
    new = importlib.import_module(f"experiments.{name}")
    d_new, d_old = os.path.join(tmp, f"{name}_new"), os.path.join(tmp, f"{name}_old")
    new.OUT = d_new
    p_new = new.collect(901, n)
    assert os.path.exists(p_new), f"{name}: routed collect wrote nothing"
    old = _old_module(name, tmp)
    old.OUT = d_old
    try:
        p_old = old.collect(901, n)
    except TypeError as e:
        assert new_sig, f"{name}: old code failed on 3.2.3 too: {e}"
        return f"{name}: PASS on {mujoco.__version__} (routed runs; old FAILS as it must: {str(e)[:60]})"
    assert not new_sig, f"{name}: old code ran on {mujoco.__version__} — control did not fire"
    assert _same(_npz(p_new), _npz(p_old)), f"{name}: routed output differs from old on 3.2.3"
    return f"{name}: PASS on {mujoco.__version__} (routed == old bit for bit)"


def main():
    with tempfile.TemporaryDirectory(prefix="j20_raytest_", dir="/data/tmp") as tmp:
        print(_case("eye_streak", "EYE_STREAK_OUT", 40, tmp))
        print(_case("find_himself", "FIND_HIMSELF_OUT", 4, tmp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
