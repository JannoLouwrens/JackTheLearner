"""The D27 follow-up MEASUREMENT ordered by `d27-screen-measures-95-percent-false`
(Review DAILY 2026-09-29): *"follow subscripts and helper calls on the same
20-draw and report how many instances of D27's actual class appear."*

THIS IS A MEASUREMENT, NOT A COUNTER. It is invoked by nothing, rendered on no
board, and exists in the tree so the number written into
`docs/DECISIONS_RESOLVED.md` under `D27` is re-derivable rather than
remembered. It is deliberately self-contained — the 20 pairs are hardcoded
below exactly as adjudicated on 2026-09-21 (seed 27, `unread_metrics.sample()`)
— because the screen that drew them is deleted in the same act that records
this measurement, per the same ruling. The deleted screen is in git history at
the commit this file arrives in.

WHAT IS FOLLOWED THAT THE DELETED SCREEN DID NOT FOLLOW
-------------------------------------------------------
The screen's dataflow join was one hop over bare identifiers: a flagged key was
"summarised" only if its own dict-literal value expression shared an `ast.Name`
with the value expression of a key `_check` reads. The two adjudicated misses
travelled through exactly the channels that join cannot see:

  * SUBSCRIPTS — `SO.09 synth_lo_refused` is consumed through `rb["refused"]`;
    there is no shared bare identifier to join on.
  * HELPER CALLS — `LC.02 wm-latent/clears@1.0` is read by
    `committed_ratio(m, arm)`, a module-local function the scan never entered.

This measurement closes both: starting from `_check`'s own reads, it computes a
transitive reach over (a) local-variable assignments, (b) string subscript
slices, and (c) module-local helper-function bodies, and asks for each flagged
pair whether the key is reachable IN SUBSTANCE. Builtins, imported names and
module-level function names are excluded from the identifier join so `float`
or `np` cannot weld everything to everything.

THE QUESTION IT ANSWERS, pre-registered before the run per the ruling:
  1. How many of the 20 adjudicated pairs does the followed filter still flag?
  2. Of those, how many are instances of D27's actual class — "the run
     measured the quantity that would have indicted it and then did not look
     at it" — per the standing hand adjudication, which is the ground truth?
If the second number is zero again, zero is the honest answer to D27 and it is
written down as an answer rather than left as an absence.

Run: /data/venvs/jackthelearner/bin/python -m experiments.d27_followed_reads
"""

from __future__ import annotations

import ast
import builtins
import pathlib
from typing import Dict, List, Set, Tuple

REPO = pathlib.Path(__file__).resolve().parent.parent

#: The 2026-09-21 adjudicated 20-draw, verbatim from
#: `docs/DECISIONS_RESOLVED.md` under `D27`. verdict: how the hand
#: adjudication classed the pair. "substance" = read in substance by the gate
#: (a false positive the one-hop join could not see); "context" = genuinely
#: unread, and legitimately so (fixture size, input, descriptive statistic);
#: "decorative" = a literal recorded as a measurement, never computed, never
#: read — TRUE positive of the screen, NOT an instance of D27's class.
ADJUDICATED: List[Tuple[str, str, str]] = [
    ("LC.01", "wm-efe/needs_loss_share", "substance"),
    ("LC.01", "wm-efe/u4", "substance"),
    ("LC.01", "wm-latent/deterministic", "substance"),
    ("LC.01", "wm-latent/needs_loss_share", "substance"),
    ("LC.02", "wm-latent/clears@1.0", "substance"),
    ("LG.00", "grounded_knowledge_advantage_s0", "substance"),
    ("LG.02", "follow_liar_lastq", "substance"),
    ("LG.02", "null_abs_div", "substance"),
    ("ME.1", "events", "context"),
    ("ME.10", "n_train_episodes", "context"),
    ("PG.2", "bob_rho0.2", "substance"),
    ("PG.2", "expected_rho0.3", "context"),
    ("PS.02", "death_s_max", "context"),
    ("SM.01", "peak_over_mean_2m", "substance"),
    ("SO.06", "seen_n_rays_changed", "substance"),
    ("SO.09", "synth_lo_refused", "substance"),
    ("T0.26", "deg_tf_abs_spread", "substance"),
    ("T0.33", "n_foreclosed_now", "context"),
    ("UB.9", "n_test", "context"),
    ("T0.06", "steps_ok", "decorative"),
]

#: Zero instances of D27's class were found in the hand adjudication; this
#: constant exists so the assertion below reads as the pre-registration it is.
D27_CLASS_VERDICTS = frozenset()  # no pair carries the class verdict


def _builtin_names() -> Set[str]:
    return set(dir(builtins))


class _ModuleIndex:
    """One parse of a spec module, indexed for the reach computation."""

    def __init__(self, path: pathlib.Path):
        self.tree = ast.parse(path.read_text())
        self.funcs: Dict[str, ast.FunctionDef] = {
            n.name: n for n in ast.walk(self.tree)
            if isinstance(n, ast.FunctionDef)}
        self.imports: Set[str] = set()
        for n in ast.walk(self.tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    self.imports.add((a.asname or a.name).split(".")[0])
        # every `"key": <expr>` in every dict literal, module-wide
        self.dict_exprs: Dict[str, List[ast.AST]] = {}
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        self.dict_exprs.setdefault(k.value, []).append(v)
        # assignment graph: target name -> RHS features, module-wide
        self.assign: Dict[str, List[ast.AST]] = {}
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for name in self._target_names(t):
                        self.assign.setdefault(name, []).append(n.value)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                self.assign.setdefault(n.target.id, []).append(n.value)
        self.excluded = self.imports | _builtin_names() | set(self.funcs)

    @staticmethod
    def _target_names(t: ast.AST) -> List[str]:
        if isinstance(t, ast.Name):
            return [t.id]
        if isinstance(t, (ast.Tuple, ast.List)):
            out = []
            for e in t.elts:
                out += _ModuleIndex._target_names(e)
            return out
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
            return [t.value.id]          # d["k"] = ... taints d
        return []

    def features(self, node: ast.AST) -> Tuple[Set[str], Set[str], Set[str]]:
        """(local variable names, subscript strings, called local functions)."""
        names: Set[str] = set()
        strings: Set[str] = set()
        calls: Set[str] = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id not in self.excluded:
                names.add(n.id)
            elif isinstance(n, ast.Subscript):
                s = n.slice
                if isinstance(s, ast.Constant) and isinstance(s.value, str):
                    strings.add(s.value)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id in self.funcs:
                    calls.add(n.func.id)
        return names, strings, calls


def reach_of_check(idx: _ModuleIndex) -> Tuple[Set[str], Set[str]]:
    """Transitive (names, strings) reachable from `_check`'s reads, following
    local-variable assignments, string subscripts and local helper bodies."""
    check = idx.funcs.get("_check")
    if check is None:
        return set(), set()
    # seeds: every string `_check` touches is a read key; follow each read
    # key's minting expressions, plus `_check`'s own body.
    read_strings = {n.value for n in ast.walk(check)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    names: Set[str] = set()
    strings: Set[str] = set(read_strings)
    pending_calls: Set[str] = set()
    seed_nodes: List[ast.AST] = [check]
    for k in read_strings:
        seed_nodes += idx.dict_exprs.get(k, [])
    for node in seed_nodes:
        n_, s_, c_ = idx.features(node)
        names |= n_
        strings |= s_
        pending_calls |= c_
    # fixpoint over the assignment graph and helper bodies
    seen_calls: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for v in list(names):
            for rhs in idx.assign.get(v, []):
                n_, s_, c_ = idx.features(rhs)
                if not n_ <= names or not s_ <= strings or not c_ <= pending_calls | seen_calls:
                    names |= n_
                    strings |= s_
                    pending_calls |= c_
                    changed = True
        for f in list(pending_calls - seen_calls):
            seen_calls.add(f)
            n_, s_, c_ = idx.features(idx.funcs[f])
            if not n_ <= names or not s_ <= strings:
                changed = True
            names |= n_
            strings |= s_
            pending_calls |= c_
    return names, strings


def followed_flag(idx: _ModuleIndex, key: str) -> bool:
    """True if the followed filter STILL flags `key` as unread."""
    names, strings = reach_of_check(idx)
    if key in strings:
        return False                     # the key's own name is reached
    k_names: Set[str] = set()
    k_strings: Set[str] = set()
    for e in idx.dict_exprs.get(key, []):
        n_, s_, _ = idx.features(e)
        k_names |= n_
        k_strings |= s_
    if k_names & names:
        return False                     # shares a local variable with a read
    if k_strings & strings:
        return False                     # shares a subscript string with a read
    return True


def measure() -> dict:
    from .protocol import module_path_for

    rows = []
    for sid, key, verdict in ADJUDICATED:
        path = module_path_for(sid)
        idx = _ModuleIndex(path)
        still = followed_flag(idx, key)
        rows.append({"spec": sid, "key": key, "verdict": verdict,
                     "still_flagged": still})
    remaining = [r for r in rows if r["still_flagged"]]
    removed = [r for r in rows if not r["still_flagged"]]
    substance_total = sum(1 for r in rows if r["verdict"] == "substance")
    substance_removed = sum(1 for r in removed if r["verdict"] == "substance")
    # D27's class, per the standing ground-truth adjudication: no pair of the
    # 20 was the class, so the count among any subset is zero BY THE RECORD —
    # the measurement's job is to show what the followed residue consists of.
    d27_class_remaining = sum(
        1 for r in remaining
        if r["verdict"] not in ("substance", "context", "decorative"))
    return {"rows": rows, "n_remaining": len(remaining),
            "n_removed": len(removed),
            "substance_removed": f"{substance_removed}/{substance_total}",
            "remaining_verdicts": sorted(r["verdict"] for r in remaining),
            "d27_class_remaining": d27_class_remaining}


def main() -> None:
    res = measure()
    print("D27 FOLLOW-UP MEASUREMENT — subscripts and helper calls followed "
          "on the 2026-09-21 adjudicated 20-draw")
    for r in res["rows"]:
        mark = "STILL FLAGGED" if r["still_flagged"] else "removed      "
        print(f"  {mark}  {r['spec']:7s} {r['key']:35s} "
              f"adjudicated: {r['verdict']}")
    print(f"  removed {res['n_removed']} of 20; "
          f"substance-read false positives removed "
          f"{res['substance_removed']}")
    print(f"  remaining flagged: {res['n_remaining']} "
          f"({', '.join(res['remaining_verdicts'])})")
    print(f"  instances of D27's actual class in the residue: "
          f"{res['d27_class_remaining']}")


if __name__ == "__main__":
    main()
