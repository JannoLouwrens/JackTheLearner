"""T0.23 — an argv the runner does not fully understand must not run anything.

THE SCAR (2026-08-11 20:08 UTC, this project, three minutes before it was
found). An agent typed a sub-command that does not exist:

    python -m experiments.run show T1.02

`show` is not a command. The runner printed `unknown spec show`, counted one
failure, and then **ran T1.02 anyway** — a `gpu<20min` spec, which built a job
and submitted it to Colab. Free-tier GPU quota was spent on a command nobody
issued. It was found only because the shell that launched it was killed and the
orphaned PID was still alive in `ps`.

The typo is not the lesson. The SHAPE is: between `cmd_run` and
`gpu.submit()` there is no further confirmation of any kind, so the argv parser
is the last gate in front of the scarcest resource this project has — and it
was written to be forgiving, skipping what it did not recognise and proceeding
with what it did. Forgiving is the correct setting for a query and the wrong
setting for a spend.

The guard is one branch in `run.main()`: if any positional token is not a spec
id, refuse the WHOLE argv with a non-zero exit. Not "refuse the bad token" —
refuse the invocation, because a partial run is exactly the failure mode. Six
properties, each able to fail on its own:

  P0  the fixture spec is still unimplemented, so "did dispatch reach cmd_run"
      stays observable without running or charging anything. Checked FIRST and
      the run bails if it is false: a fixture that quietly starts doing work is
      this same bug one level up.
  P1  the malformed argv exits non-zero AND says so — the refusal line must be
      present, not merely the exit code (see the lock note below).
  P2  the malformed argv never reaches the spec (no `[<spec>]` line).
  P3  a read-only command (`status`) is untouched — exit 0.
  P4  a bare, well-formed spec id is NOT refused: exit 0 and no refusal line.
      Without P4 a guard that refuses everything would pass this spec.
  P5  a good spec id BESIDE a bad token is refused too, by the same line. This
      is the "no partial run" property, and it is the one the bug violated.

WHY P4 CANNOT ASSERT DISPATCH, stated plainly rather than quietly weakened.
`_exclusive` is process-wide: this spec runs while holding the runner lock, so
every sub-invocation it makes prints `Another run holds …` and exits **0**
without dispatching. So `rc == 0` alone cannot distinguish "ran" from "declined
to run", and P1/P5 assert the refusal LINE for exactly that reason — an exit
code that a lock can manufacture is not evidence. What "reaching `cmd_run`"
looks like is established instead by the control, which calls it directly with
no lock in the way and gets the `[<spec>]` line.

THE CONTROL is the pre-guard dispatch replayed verbatim — `cmd_run(ledger,
argv)`, literally the line `main()` used to end on — against the same malformed
argv. It MUST reach the spec. A control that also refuses would mean the
fixture argv is harmless and this spec measures nothing.

Everything runs against a TEMPORARY ledger or a spec with no implementation;
this test never writes the real record and never submits a job.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

#: A registered spec that (a) has no implementation, so dispatch is observable
#: without work, and (b) is `cpu`, so the GPU freshness pre-check cannot
#: short-circuit `main()` before the branch under test is reached — a dirty
#: working tree is the NORMAL state of the iteration that runs this.
FIXTURE_SPEC = "T6.02"
BAD_TOKEN = "show"          # the exact token that cost the quota

#: The guard's own words. Asserted rather than assumed, because the runner lock
#: can produce a clean exit code without running anything (see the docstring).
REFUSAL = "Refusing to run: unrecognised argument(s)"
LOCK_SKIP = "Another run holds"


def _cli(argv: list[str]) -> tuple[int, str]:
    """Run the SHIPPED command line, not a restatement of it (T0.16's rule)."""
    p = subprocess.run([sys.executable, "-m", "experiments.run", *argv],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _reached_spec(out: str) -> bool:
    """`cmd_run` announces every spec it dispatches as `[<id>] …`."""
    return f"[{FIXTURE_SPEC}]" in out


def _experiment(seed: int) -> dict:
    from ..run import _module_for

    fixture_unimplemented = _module_for(FIXTURE_SPEC) is None
    m = {"fixture_unimplemented": fixture_unimplemented,
         "fixture_spec": FIXTURE_SPEC}
    if not fixture_unimplemented:
        # Bail before running anything: with an implementation present, every
        # property below would dispatch real work into the real ledger.
        m["bail"] = ("fixture spec gained an implementation — move the fixture "
                     "to another unimplemented cpu spec before trusting P1-P5")
        return m

    rc_bad, out_bad = _cli([BAD_TOKEN, FIXTURE_SPEC])
    rc_ro, _ = _cli(["status"])
    rc_good, out_good = _cli([FIXTURE_SPEC])
    rc_mixed, out_mixed = _cli([FIXTURE_SPEC, "T1.O2"])   # letter O, a real typo

    m.update({
        "bad_argv_refused": rc_bad != 0 and REFUSAL in out_bad,
        "bad_argv_never_dispatched": not _reached_spec(out_bad),
        "readonly_still_works": rc_ro == 0,
        "good_argv_not_refused": rc_good == 0 and REFUSAL not in out_good,
        "mixed_argv_refused": (rc_mixed != 0 and REFUSAL in out_mixed
                               and not _reached_spec(out_mixed)),
        # Reported, not gated: which of the two legal outcomes the good argv
        # got. `True` here means the runner lock was held by this very spec.
        "good_argv_lock_skipped": LOCK_SKIP in out_good,
        "good_argv_dispatched": _reached_spec(out_good),
        "rc_bad": rc_bad, "rc_readonly": rc_ro,
        "rc_good": rc_good, "rc_mixed": rc_mixed,
    })
    return m


def _control(seed: int) -> dict:
    """The pre-2026-08-11 dispatch, verbatim: `cmd_run(ledger, args.spec)`.

    It is given the same malformed argv and a throwaway ledger. It must reach
    the fixture spec — that reach IS the historical defect.
    """
    import io
    from contextlib import redirect_stdout

    from ..run import cmd_run

    with tempfile.TemporaryDirectory() as td:
        led = Ledger(Path(td) / "ledger.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_run(led, [BAD_TOKEN, FIXTURE_SPEC])
        out = buf.getvalue()

    return {"prefix_reached_spec": _reached_spec(out),
            "prefix_saw_unknown_token": f"unknown spec {BAD_TOKEN}" in out,
            "prefix_rc": rc}


_PROPS = ("fixture_unimplemented", "bad_argv_refused",
          "bad_argv_never_dispatched", "readonly_still_works",
          "good_argv_not_refused", "mixed_argv_refused")


def _check(m: dict, c: dict) -> bool:
    failed = [k for k in _PROPS if not m.get(k, False)]
    m["properties_failed"] = len(failed)
    m["failed_properties"] = failed

    # The control must have DONE the thing the guard now forbids. `.get(k, False)`
    # so an empty control reads as "the old dispatch was already safe", which is
    # false — a control this gate does not read is a control that is not there.
    control_is_a_decoy = (c.get("prefix_reached_spec", False) is True
                          and c.get("prefix_saw_unknown_token", False) is True)
    m["control_reached_spec"] = control_is_a_decoy
    return not failed and control_is_a_decoy


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.23"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
