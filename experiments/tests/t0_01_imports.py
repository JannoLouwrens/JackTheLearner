"""T0.01 — every live module imports cleanly, with no side effects.

An import that starts training, opens a window, or downloads a model makes every
later timing and memory measurement meaningless.

## WHY THIS SPEC DECLARES `IMPL_DEPS` (armed 2026-09-27, strengthen-only)

This certificate's whole subject is the bytes of thirteen root modules, and for
49 days it hashed none of them. `impl_sha` covered `t0_01_imports.py` alone, so
adding a model download to `Persistence.py` — the precise failure this spec
exists to catch — left T0.01's PASS standing and `run stale` silent. That is
the PG.6 scar exactly (`protocol.impl_sha_of`: *"a certificate about a world
invalidated by a change to the world, undetectable by design"*), and T0.01 was
its largest remaining instance: **five root modules of Jack — `AudioListener`,
`InnerMonologue`, `Persistence`, `Personality`, `TaskManager`, 146,391 bytes —
were named in NO spec's `IMPL_DEPS` at all**, and all five are in `LIVE_MODULES`
below (measurement, builder, 2026-09-27; full table in `docs/LOOP_JOURNAL.md`).

The declaration has to be a literal (`protocol.impl_deps_of` reads it statically
with `ast.literal_eval`, deliberately, so that reading a sha never imports the
module), so it CANNOT be computed from `LIVE_MODULES` — which means the two
lists can drift, and a list that drifts silently is how this hole was dug in the
first place. `deps_cover_live_modules` closes that: the run measures whether
every module it imported is also a path it hashed, and `_check` gates it. Adding
a module to `LIVE_MODULES` without adding its file here now FAILS the spec
rather than quietly shrinking what the certificate covers.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

LIVE_MODULES = [
    "UnifiedBrain", "VirtualWorld", "TaskManager", "Persistence", "EmotionalState",
    "Personality", "MovementMoodCoupling", "InnerMonologue", "SymbolicCalculator",
    "AlphaGeometryLoop", "MoCapLoader", "TrainingPipeline", "AudioListener",
]

#: The bytes this certificate is ABOUT. Kept in lock-step with `LIVE_MODULES`
#: by the `deps_cover_live_modules` conjunct, never by care.
IMPL_DEPS = [
    "UnifiedBrain.py", "VirtualWorld.py", "TaskManager.py", "Persistence.py",
    "EmotionalState.py", "Personality.py", "MovementMoodCoupling.py",
    "InnerMonologue.py", "SymbolicCalculator.py", "AlphaGeometryLoop.py",
    "MoCapLoader.py", "TrainingPipeline.py", "AudioListener.py",
]


def _declared_impl_deps() -> list:
    """`IMPL_DEPS` as the sha reader sees it — statically, from this file.

    Read through the AST rather than off the module global on purpose: the
    value that lands in `impl_sha` is the one `protocol.impl_deps_of` parses out
    of the source, so a conjunct asserting coverage must assert it about THAT
    value. A conjunct reading the live global would pass while the hash read
    something else — the two-code-paths failure `impl_sha_of`'s docstring
    records, in miniature.
    """
    tree = ast.parse(Path(__file__).read_bytes())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "IMPL_DEPS"
                for t in node.targets):
            return list(ast.literal_eval(node.value))
    return []


#: An import slower than this is doing WORK (a download, a dataset scan).
SLOW_LINE_S = 5.0
#: Ruling 132 (S21): each module is timed in a fresh process up to this many
#: times and the FASTEST is judged — a busy shared box slows some tries, a
#: module that does work at import slows every one. Tries stop at the first
#: one under the line: the fastest can then only be under it too, so the
#: verdict is the one three tries would give, at a third of the wall time.
TRIES = 3

_TIMER = ("import sys,time; sys.path.insert(0, sys.argv[1]); "
          "t=time.perf_counter(); __import__(sys.argv[2]); "
          "print('T0_01_DT', time.perf_counter()-t)")


def time_imports(modules, path, line=SLOW_LINE_S, tries=TRIES):
    """Per module: the import times of each fresh-process try, or the error.

    Returns {name: {"times": [s, ...], "error": str|None}}.
    """
    import subprocess
    out = {}
    for name in modules:
        times, error = [], None
        for _ in range(tries):
            r = subprocess.run([sys.executable, "-c", _TIMER, str(path), name],
                               capture_output=True, text=True, cwd=str(path))
            dts = [l.split()[1] for l in r.stdout.splitlines()
                   if l.startswith("T0_01_DT ")]
            if r.returncode != 0 or not dts:
                tail = (r.stderr.strip().splitlines() or ["exit %d" % r.returncode])[-1]
                error = tail.split(":")[0][:80]
                break
            times.append(float(dts[-1]))
            if times[-1] <= line:
                break
        out[name] = {"times": times, "error": error}
    return out


def _experiment(seed: int) -> dict:
    import os
    load_1m = os.getloadavg()[0]
    res = time_imports(LIVE_MODULES, REPO)
    failed = [f"{n}: {r['error']}" for n, r in res.items() if r["error"]]
    ok = len(LIVE_MODULES) - len(failed)
    # A slow import is a side effect: model download, dataset scan, or worse.
    slow = [f"{n}:" + "/".join(f"{t:.1f}" for t in r["times"]) + "s"
            for n, r in res.items()
            if r["times"] and min(r["times"]) > SLOW_LINE_S]
    retried = [f"{n}:" + "/".join(f"{t:.1f}" for t in r["times"]) + "s"
               for n, r in res.items() if len(r["times"]) > 1]
    # Does the certificate hash every module it just judged? A module imported
    # here but absent from IMPL_DEPS is a claim about bytes nothing pins.
    declared = set(_declared_impl_deps())
    unhashed = sorted(f"{n}.py" for n in LIVE_MODULES if f"{n}.py" not in declared)
    # And the reverse: a declared path that does not resolve hashes as the
    # literal string `missing:<path>` (impl_sha_of), i.e. a permanent mismatch
    # rather than coverage. Report it as the absence it is.
    unresolved = sorted(p for p in declared if not (REPO / p).is_file())
    return {
        "modules_imported": ok,
        "modules_total": len(LIVE_MODULES),
        "failed": ";".join(failed) or "none",
        "slow_imports": ";".join(slow) or "none",
        "retried_imports": ";".join(retried) or "none",
        "fastest_import_max_s": round(max((min(r["times"]) for r in res.values()
                                           if r["times"]), default=0.0), 2),
        "load_1m": round(load_1m, 2),
        "deps_cover_live_modules": float(not unhashed and not unresolved),
        "live_modules_unhashed": ";".join(unhashed) or "none",
        "declared_deps_unresolved": ";".join(unresolved) or "none",
        "declared_deps": len(declared),
    }


def _check(m: dict, _c: dict) -> bool:
    return (m["modules_imported"] == m["modules_total"]
            and m["slow_imports"] == "none"
            # armed 2026-09-27, strengthen-only: it can only ever subtract a
            # PASS, and it subtracts exactly the case where this spec's own
            # subject drifted out from under its hash.
            and m["deps_cover_live_modules"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.01"], _experiment, _check, ledger=ledger)
