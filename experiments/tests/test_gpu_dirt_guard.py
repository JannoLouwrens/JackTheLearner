"""S7(1) planted test: the GPU submit guard refuses only dirt the VM could load.

(a) Foreign dirt (dashboard files, the status script) is NOT fatal and is named.
(b) Planted: a dirty file under experiments/, and a top-level module that
    experiment code imports, MUST still refuse.
(c) Planted in a scratch root: a module named by experiments/ is fatal, an
    unnamed sibling is foreign — the closure is read from the code, not a list.
(d) Control that must fail: the old tree-wide predicate (`offending_dirt`)
    refuses the dashboard line — else (a) measured nothing.
"""
import tempfile
from pathlib import Path

from experiments.gpu import offending_dirt, split_dirt

FOREIGN = [" M dashboard/page.html", " M dashboard/vitals.json",
           " M scripts/dashboard_status.py"]
FATAL = [" M experiments/jack_env.py", " M experiments/gpu.py",
         " M TrainingPipeline.py"]
OUTPUTS = [" M experiments/ledger.json", " M CHECKLIST.md"]


def main():
    fatal, foreign = split_dirt(FOREIGN + FATAL + OUTPUTS)
    assert fatal == FATAL, f"(b) planted imported dirt must refuse: {fatal}"
    assert foreign == FOREIGN, f"(a) foreign dirt must be named: {foreign}"
    with tempfile.TemporaryDirectory(dir="/data") as d:
        root = Path(d)
        (root / "experiments").mkdir()
        (root / "experiments" / "x.py").write_text("import foo_mod\n")
        f, g = split_dirt([" M foo_mod.py", " M bar_mod.py"], root)
        assert f == [" M foo_mod.py"] and g == [" M bar_mod.py"], (f, g)
    split_dirt([], None)  # reset the cached corpus to the real tree
    assert " M dashboard/page.html" in offending_dirt(FOREIGN), "(d) control"
    print("PASS test_gpu_dirt_guard: 4/4")


if __name__ == "__main__":
    main()
