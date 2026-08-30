"""T0.30 — the regression gate cannot demote the certificates it re-runs.

THE SCAR IS THIS ITERATION'S OWN INHERITANCE, not an audit finding.

2026-08-30 09:19 UTC, commit `7966524`. The builder changed
`experiments/champions.py`, ran `python -m experiments.run --gate` to check for
regressions, and committed afterwards. Correct instinct, fatal order. All ten
specs the gate re-ran recorded `e9bd4a0+dirty`, and for `T0.08` and `T0.09` the
recorded `impl_sha` reconstructed from no committed blob, so both PASSes became
DIRTY STAMPS. `T0.09` is a dependency of 36 specs. For the next three hours
`run blocked` printed a phantom at the top of the project —

    T0.09 = PASS but STALE — re-run it  frees 36  (blocks 37)

— above the real top blocker (`T2.01`, frees 35), and clearing it cost two
re-runs and a second Colab T4 round-trip. **Ten tests passed and the ladder got
worse.**

THE CLASS, which is why this is a spec and not a fix. A single-spec run from a
dirty tree merely fails to certify, and that is the ordinary case: `t0_23`'s own
fixture note says a dirty working tree is the NORMAL state of the iteration that
runs it. A GATE run is different in kind — it re-runs rows that ALREADY hold
clean stamps, so from a dirty tree its expected effect on the record is negative
by construction, and `blocked_by` propagates the demotion to every dependent.
The same event is on record twice under other names — `T2.00`'s `08444b2+dirty`
(998-second re-run, 47 specs blocked, caused by an uncommitted
`LOOP_JOURNAL.md`) and `T0.25`'s `1ddcd27+dirty` — and both were repaired as
incidents. Neither was repaired as a class, which is why it happened a third
time.

NOTHING HERE WEAKENS A STAMP. `+dirty` fires on exactly the condition it fired
on before; `protocol.gate_precondition` only refuses to VOLUNTEER for it, and
`--dirty-ok` keeps "does my uncommitted change break anything" available at the
cost of saying so out loud.

P6-P8 run the SHIPPED command line (T0.16's rule) in a scratch `git clone`, not
a restatement of it, because the property under test is an ORDERING inside
`main()` — the refusal must land before `_exclusive` takes the global lock and
before `cmd_run` dispatches anything. A pure-function battery cannot see that.
The clone's ledger is replaced with a single PASS for an UNIMPLEMENTED spec
(`T6.02`, borrowed from `t0_23`), so a gate that wrongly proceeds dispatches
observably and costs nothing.

THE CONTROL is the gate as it stood before 2026-08-30, reconstructed BY
DELETION rather than paraphrase (T0.08 property 5): a second clone whose
`protocol.py` carries an appended `gate_precondition(*_, **__) -> ""`, so the
guard is present, imported and called, and simply never fires. The append
itself dirties that clone — which is precisely the condition under test. It
must let `--gate` start from a modified tree.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..protocol import (DOC_OUTPUTS, GATE_DIRTY_FLAG, Ledger, RUNNER_OUTPUTS,
                        code_dirt, gate_precondition, run_spec)
from ..registry import BY_ID

SPEC_ID = "T0.30"
N_PROPERTIES = 8.0
REPO = Path(__file__).resolve().parents[2]

#: The file whose uncommitted edit caused the real event.
SCAR_PATH = "experiments/champions.py"

#: A registered spec with no implementation, so "the gate dispatched" is
#: observable without doing any work. Same fixture and same reason as `t0_23`.
FIXTURE_SPEC = "T6.02"

#: The guard's own words, asserted rather than assumed.
REFUSAL = "Refusing to gate:"

#: Reconstruct-by-deletion: the guard stays imported and called, and returns
#: the "may proceed" value unconditionally. Appended to the clone's module, so
#: the redefinition wins at `from .protocol import gate_precondition` time.
LEGACY_APPEND = (
    "\n\n# T0.30 CONTROL — the gate as it stood before 2026-08-30.\n"
    "def gate_precondition(*_a, **_k):\n"
    "    return \"\"\n"
)


def _clone(dst: Path) -> bool:
    """A scratch checkout of the CURRENT WORKING TREE, committed, clean.

    `--depth 1` over the `file://` transport, which costs ~0.8 s and 24 MB
    against a 97 MB object store. NOT `--local`: that hardlinks, and the
    scratch root is on `/data` while the repo is on `/home`, so it dies with
    *"Invalid cross-device link"*. It then copies every code file this tree
    has modified and
    commits them INSIDE the clone, for a reason that is not convenience: a
    clone of HEAD alone could only ever test the guard as already shipped, so
    the one spec in the ladder about running the gate before committing would
    be the one spec you cannot run before committing. `git add -A` is banned in
    the shared tree and is correct here — this repository is three seconds old
    and nobody else can be working in it.

    Returns False if git is unavailable or the clone is unusable; the caller
    then records P6-P8 as UNEVALUATED rather than as passed.
    """
    p = subprocess.run(["git", "clone", "--quiet", "--depth", "1",
                        f"file://{REPO}", str(dst)],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or not (dst / "experiments" / "run.py").exists():
        return False
    porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True,
                               timeout=30).stdout.splitlines()
    for rel in code_dirt(porcelain):
        src, dest = REPO / rel, dst / rel
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        elif dest.exists():
            dest.unlink()
    subprocess.run(["git", "add", "-A"], cwd=dst, capture_output=True, timeout=60)
    subprocess.run(["git", "-c", "user.name=t030", "-c", "user.email=t030@local",
                    "commit", "--quiet", "--allow-empty", "-m", "T0.30 fixture"],
                   cwd=dst, capture_output=True, timeout=60)
    if subprocess.run(["git", "status", "--porcelain"], cwd=dst,
                      capture_output=True, text=True, timeout=30).stdout.strip():
        return False        # the fixture must START clean or P7 measures nothing
    # One PASS, for a spec with no implementation: the gate set is then exactly
    # [T6.02] and a gate that wrongly proceeds is visible and free.
    stub = {"_comment": "T0.30 scratch fixture — not a record of anything.",
            "results": {FIXTURE_SPEC: {"amended": [], "attempt": 1,
                                       "commit": "fixture", "history": [],
                                       "metrics": {}, "control_metrics": {},
                                       "ran_at": "2026-08-30T00:00:00",
                                       "spec_id": FIXTURE_SPEC,
                                       "status": "PASS"}}}
    (dst / "experiments" / "ledger.json").write_text(json.dumps(stub, indent=2))
    return True


def _gate(cwd: Path, extra=()) -> tuple:
    """Run the shipped `--gate` command line inside a clone."""
    p = subprocess.run([sys.executable, "-m", "experiments.run", "--gate", *extra],
                       cwd=cwd, capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _dispatched(out: str) -> bool:
    """`cmd_run` announces every spec it dispatches as `[<id>] …`."""
    return f"[{FIXTURE_SPEC}]" in out


def _probe(legacy: bool) -> dict:
    precond = (lambda *_a, **_k: "") if legacy else gate_precondition
    failed = []

    # --- P1: the recorded event. An uncommitted code file refuses, by name.
    scar = [f" M {SCAR_PATH}"]
    r1 = precond(scar, at_risk=89)
    if not (code_dirt(scar) == [SCAR_PATH] and r1 and SCAR_PATH in r1):
        failed.append("p1_uncommitted_code_refuses")

    # --- P2: the gate must not deadlock against files the runner writes.
    # This is the `gpu_budget.json` failure one surface over: an evidence log
    # that invalidates the evidence. Every excluded path, not a sample.
    outputs = [f" M {p}" for p in (RUNNER_OUTPUTS + DOC_OUTPUTS)]
    if code_dirt(outputs) or precond(outputs, at_risk=89):
        failed.append("p2_runner_outputs_do_not_refuse")

    # --- P3: a clean tree gates. A guard that refuses everything is not a
    # guard, it is an outage.
    if precond([], at_risk=89) or precond([""], at_risk=89):
        failed.append("p3_clean_tree_gates")

    # --- P4: the opt-in works AND is not the default. The default is read off
    # the shipped signature, never asserted: a flag that silently defaults to
    # True is the whole guard undone, and prose cannot catch it.
    import inspect
    default_ok = inspect.signature(gate_precondition).parameters["dirty_ok"].default
    if precond(scar, at_risk=89, dirty_ok=True) or default_ok is not False:
        failed.append("p4_dirty_ok_is_an_opt_in")

    # --- P5: the refusal is informative — it reports the exposure it prevented.
    r5 = precond(scar, at_risk=89)
    if not (r5 and "89" in r5 and GATE_DIRTY_FLAG in r5):
        failed.append("p5_refusal_reports_exposure")

    # --- P6/P7/P8: the shipped command line, in a scratch clone.
    root = Path(tempfile.mkdtemp(prefix="t030-", dir="/data"))
    clone_ok = False
    try:
        work = root / "clone"
        clone_ok = _clone(work)
        if not clone_ok:
            failed += ["p6_dirty_gate_dispatches_nothing",
                       "p7_clean_gate_still_runs",
                       "p8_dirty_ok_gates_anyway"]
        else:
            if legacy:
                # Reconstruct the pre-2026-08-30 gate by deletion. The append
                # is itself an uncommitted code edit, so the tree is dirty for
                # the same reason the real event's tree was.
                with open(work / "experiments" / "protocol.py", "a") as fh:
                    fh.write(LEGACY_APPEND)

            # P7 first, while the only modification is `ledger.json` — a
            # RUNNER_OUTPUT, so this doubles as an end-to-end proof of P2.
            rc7, out7 = _gate(work)
            if REFUSAL in out7 or rc7 != 0:
                failed.append("p7_clean_gate_still_runs")

            # Now dirty it the way the builder did: edit a code file.
            with open(work / SCAR_PATH, "a") as fh:
                fh.write("\n# T0.30 fixture edit\n")

            rc6, out6 = _gate(work)
            if not (rc6 != 0 and REFUSAL in out6 and not _dispatched(out6)):
                failed.append("p6_dirty_gate_dispatches_nothing")

            rc8, out8 = _gate(work, [GATE_DIRTY_FLAG])
            if REFUSAL in out8 or rc8 != 0:
                failed.append("p8_dirty_ok_gates_anyway")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    return {
        "properties_checked": N_PROPERTIES,
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "clone_built": clone_ok,
    }


def _experiment(seed: int) -> dict:
    return _probe(legacy=False)


def _control(seed: int) -> dict:
    """The gate as it stood before 2026-08-30, kept executable.

    One hole, and it is an absence rather than a bug: nothing asked whether the
    tree was clean before re-running ninety certificates over it. It must fail
    P1, P5 and P6 — and it must still pass P2, P3, P7 and P8, which is what
    makes it a control rather than a broken import. The old gate ran fine; it
    simply could not tell a certification sweep from a demotion sweep.
    """
    return _probe(legacy=True)


def _check(m: dict, c: dict) -> bool:
    # Every property RAN and every property held. Gating on
    # `properties_failed == 0` alone lets a battery that stopped early read as
    # clean — T0.13's own first bug, and every T0.1x/T0.2x battery since. The
    # clone is part of the claim: if it could not be built, P6-P8 are
    # unevaluated, not passed, and they are already in `failed_names`.
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and bool(m.get("clone_built"))
                        and c["properties_checked"] == N_PROPERTIES)
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p1_uncommitted_code_refuses",
                           "p5_refusal_reports_exposure",
                           "p6_dirty_gate_dispatches_nothing"} <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
