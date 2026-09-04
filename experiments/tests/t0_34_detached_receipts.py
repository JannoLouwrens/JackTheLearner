"""T0.34 — the detached lane writes its own receipts.

T0.33 metered the runner's children and REFUSED cpu<48h with a routing
reason, naming this gap in its own docstring: the lane the refusal routes to
(`scripts/launch_detached.sh`) kept no accounts. The scar is on the ledger —
LC.03 v2 spent ~190 core-hours over 2.6 days through this lane on a 4-core
box with paying tenants, and no meter saw a second of it while it ran. The
claim: a detached launch is ADMITTED against the same day ledger before it
detaches, and BILLS its measured wall clock incrementally as it runs.

Two design choices carry the claim, both gated below rather than asserted:

  INCREMENTAL, NOT LUMP-SUM. A lump charge at exit would dump a multi-day
  child's whole life into its exit day (57 h into a 16 h bucket — the
  day-collapse disease T0.33's control exhibits, one lane over) and would
  lose EVERYTHING under SIGKILL. Heartbeat billing bounds both: every day is
  charged the seconds the child actually spent inside it, and a group-kill
  loses at most one heartbeat.

  EST-FREE ADMISSION. A cpu<48h child cannot pre-fit a 16 h day, so the gate
  asks only whether today has headroom and the box is calm. An exhausted or
  overloaded day refuses NEW launches; a running child is never killed by
  accounting — the ceiling binds it through `charge`'s overrun marks
  (T0.33's observed-overrun posture, verbatim).

Properties, each independently checkable:

  1. DAY-SPLIT EXACT: a midnight-straddling interval lands in both day
     buckets with the exact seconds each side of midnight; a 3-day interval
     lands in every day it touched; the bucket sum equals the interval
     length (conservation); a zero-length interval moves nothing.
  2. ADMISSION: a fresh day admits; a drained day refuses with the reason
     naming the day budget, and refusal moves ZERO bytes on the budget file;
     injected load above LOAD_CEILING refuses independently of the budget.
  3. THE WRAPPER BILLS WHILE THE CHILD IS ALIVE — read mid-run, not after
     (the T0.33 control's alive-while-failing idiom, inverted: a meter that
     only writes at exit is the lump-sum disease wearing a heartbeat).
  4. SIGKILL OF THE GROUP UNDERCHARGES BY AT MOST ONE HEARTBEAT (plus
     scheduling slack), and the payload dies with the wrapper.
  5. EXIT CODES PROPAGATE through the wrapper — accounting must be invisible
     to the caller's control flow.
  6. WIRING, scanned on the live launcher source: `admit` is called BEFORE
     the setsid line, the refusal path exits nonzero before anything
     detaches, and the setsid line runs the payload under `wrap`.
  7. END TO END: a real launch through `scripts/launch_detached.sh` (against
     a temp ledger via JACK_CPU_BUDGET) produces a receipt whose total
     covers the payload's life. This exercises the shipped path — admit,
     setsid, wrap, heartbeat, final bill — not a reimplementation of it.
  8. DISJOINT CHARGES (68th audit B1/B2). This spec extended the day meter
     into the lane T0.33's property 4 declares merely ROUTED ("the detached
     lane, which this meter honestly does not see") — and per the 2026-09-04
     lesson, extending an instrument into an excluded lane re-opens the
     exclusion: T0.33's property 5 (foreclosure) also compares against the
     fresh ceiling only, so neither certificate could see the seam between
     the two meters. The seam was real: the wrapper billed the whole tree's
     wall clock AND every `run_spec` grandchild billed itself — 1.7x
     overcharge, measured live, ~35 minutes from foreclosing 53 of 152 CPU
     specs on 2.4% genuinely spent. The property: a `charge_cpu_child`-shaped
     descendant run under `wrap` against a temp budget bills NOTHING itself
     (the wrapper owns the tree), and the day's total equals the tree's true
     wall clock within one heartbeat plus slack — disjointness measured end
     to end through the composed path, not per component.

Control (registry): a lump-sum accountant billing the whole interval to the
EXIT day must FAIL the day-split property (yesterday's bucket empty, the
exit day holding everything) while its TOTAL stays exact — alive while
failing, so a broken-everywhere control cannot pass vacuously. A second
control carries property 8's burden: the same composed run with the
ownership marker deliberately stripped inside the descendant must land the
descendant's charge (alive) and push the day total past the wall-clock bound
(double-bills) — the pre-fix accountant, reproduced through the real path,
must fail the disjointness bound.

VOID lane (T0.33's, verbatim): a live 1-minute load above LOAD_CEILING is a
co-tenant condition — the heartbeat timing asserts and the admission
properties cannot be honestly attributed. Instrument fault, not refutation.

Scope, honest: dispatch.sh's GPU watchers do not route through
launch_detached.sh and are deliberately unbilled (remote waiting is not box
CPU); hand-invoked modules are the owner's lane. A wrapper killed alone
(plain kill -9 on its pid, not the group) stops the meter while the payload
runs on — the leftover check still sees the orphan; kill the GROUP.

Uses temp budget files for every property; the end-to-end launch also runs
against a temp ledger. The real file's first detached receipt arrives with
the next genuine launch — this test must never touch the real accounting.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..cpu_budget import (CPU_DAY_CEILING_S, ENV_HEARTBEAT, ENV_OVERRIDE,
                          ENV_WRAPPED, HEARTBEAT_S, LOAD_CEILING, CpuBudget,
                          _loadavg, admit_detached, bill_interval)
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

IMPL_DEPS = ["experiments/cpu_budget.py", "scripts/launch_detached.sh"]

TEST_HB_S = 0.2          # heartbeat the subprocess properties run under
KILL_AFTER_S = 1.2       # when the SIGKILL property kills the group
SLACK_S = 0.4            # scheduling slack granted to timing asserts


def _label_total(path: Path, label: str) -> float:
    """A label's charges summed across every day bucket — receipts must be
    found wherever midnight put them."""
    if not path.exists():
        return 0.0
    total = 0.0
    for bucket in CpuBudget(path).data["days"].values():
        total += bucket["by_spec"].get(label, 0.0)
    return total


def _wrap_cmd(label: str, *payload: str) -> list:
    return [sys.executable, "-m", "experiments.cpu_budget", "wrap",
            label, *payload]


def _experiment(seed: int) -> dict:
    live_load = _loadavg()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ── 1. day-split arithmetic, on synthetic epochs ──────────────────
        # midnight of a fixed, DST-quiet date on this box's clock
        base = time.mktime((2026, 9, 1, 0, 0, 0, 0, 0, -1))
        p1 = td / "split.json"
        bill_interval("t0_34_split", base - 60.0, base + 120.0, path=p1)
        b = CpuBudget(p1)
        straddle_ok = (b.used_s("2026-08-31") == 60.0
                       and b.used_s("2026-09-01") == 120.0)

        p2 = td / "multi.json"
        t0, t1 = base + 3600.0, base + 3 * 86400.0 + 7200.0
        bill_interval("t0_34_multi", t0, t1, path=p2)
        mb = CpuBudget(p2)
        days = {d: mb.used_s(d) for d in
                ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04")}
        multi_day_ok = (days["2026-09-01"] == 82800.0
                        and days["2026-09-02"] == 86400.0
                        and days["2026-09-03"] == 86400.0
                        and days["2026-09-04"] == 7200.0)
        conservation_ok = sum(days.values()) == t1 - t0

        p3 = td / "zero.json"
        bill_interval("t0_34_zero", base, base, path=p3)
        zero_len_ok = not p3.exists() or CpuBudget(p3).data["days"] == {}

        # ── 2. admission ──────────────────────────────────────────────────
        fresh = td / "fresh.json"
        d = admit_detached("t0_34_probe", path=fresh, loadavg=0.0)
        fresh_admits = d.admitted

        drained = td / "drained.json"
        CpuBudget(drained).charge("t0_34_drain", CPU_DAY_CEILING_S)
        before = drained.read_bytes()
        d = admit_detached("t0_34_probe", path=drained, loadavg=0.0)
        exhausted_refused = ((not d.admitted)
                            and "day budget" in d.reason
                            and drained.read_bytes() == before)

        d = admit_detached("t0_34_probe", path=td / "calm.json",
                           loadavg=LOAD_CEILING + 1.0)
        load_refused = (not d.admitted) and "load" in d.reason

        # ── environment the subprocess properties share ───────────────────
        def env_for(budget_file: Path) -> dict:
            env = dict(os.environ)
            env[ENV_OVERRIDE] = str(budget_file)
            env[ENV_HEARTBEAT] = str(TEST_HB_S)
            env.pop("JACK_AWAITING_SPEC", None)  # a probe owes no result
            # If THIS test runs under a real detached wrapper, the ownership
            # marker must not leak into the temp-budget universes below — an
            # inner wrapper that defers to the outer owner would bill nothing
            # and break every heartbeat property for the wrong reason.
            env.pop(ENV_WRAPPED, None)
            return env

        # ── 5. exit codes propagate ───────────────────────────────────────
        rc = subprocess.run(_wrap_cmd("t0_34_rc", "/bin/sh", "-c", "exit 7"),
                            cwd=REPO, env=env_for(td / "rc.json"),
                            capture_output=True, timeout=60).returncode
        rc_propagates = rc == 7

        # ── 3. bills while alive ──────────────────────────────────────────
        hb_file = td / "hb.json"
        proc = subprocess.Popen(
            _wrap_cmd("t0_34_hb", "/bin/sh", "-c", "sleep 3"),
            cwd=REPO, env=env_for(hb_file))
        alive_billing = False
        deadline = time.time() + 30.0
        while proc.poll() is None and time.time() < deadline:
            if _label_total(hb_file, "t0_34_hb") > 0.0:
                alive_billing = proc.poll() is None  # read WHILE running
                break
            time.sleep(0.05)
        proc.wait(timeout=60)
        hb_total = _label_total(hb_file, "t0_34_hb")
        hb_covers_life = hb_total >= 2.5  # payload slept 3s; final bill lands

        # ── 4. group SIGKILL bounded by one heartbeat ─────────────────────
        kill_file = td / "kill.json"
        marker = "sleep 30.417"
        kproc = subprocess.Popen(
            _wrap_cmd("t0_34_kill", "/bin/sh", "-c", marker),
            cwd=REPO, env=env_for(kill_file), start_new_session=True)
        k0 = time.time()
        time.sleep(KILL_AFTER_S)
        elapsed_at_kill = time.time() - k0
        os.killpg(kproc.pid, signal.SIGKILL)
        kproc.wait(timeout=30)
        time.sleep(0.3)
        billed = _label_total(kill_file, "t0_34_kill")
        sigkill_bounded = billed >= elapsed_at_kill - TEST_HB_S - SLACK_S
        payload_died = subprocess.run(
            ["pgrep", "-f", marker], capture_output=True).returncode != 0

        # ── 6. wiring, scanned on the live launcher ───────────────────────
        src = (REPO / "scripts" / "launch_detached.sh").read_text()
        admit_at = src.find("cpu_budget admit")
        m = re.search(r"^setsid .*$", src, re.M)
        setsid_at, setsid_line = (m.start(), m.group(0)) if m else (-1, "")
        wiring_ok = (0 <= admit_at < setsid_at
                     and "cpu_budget wrap" in setsid_line
                     and "exit 3" in src[admit_at:setsid_at])

        # ── 7. end to end through the shipped launcher ────────────────────
        e2e_file = td / "e2e.json"
        log = td / "probe.log"
        r = subprocess.run(
            ["sh", str(REPO / "scripts" / "launch_detached.sh"), str(log),
             "/bin/sh", "-c", "sleep 17"],
            cwd=REPO, env=env_for(e2e_file),
            capture_output=True, text=True, timeout=90)
        launcher_ok = r.returncode == 0 and "ALIVE" in r.stdout
        label = "detached:probe.log"
        deadline = time.time() + 40.0
        e2e_total = 0.0
        while time.time() < deadline:
            e2e_total = _label_total(e2e_file, label)
            if e2e_total >= 15.0:
                break
            time.sleep(0.5)
        live_receipt = launcher_ok and e2e_total >= 15.0

        # ── 8. disjoint charges through the composed path ─────────────────
        # A run_spec-shaped descendant: sleeps, then debits the runner lane
        # the way run.py:_bill_cpu does (charge_cpu_child, big fake seconds
        # so a double-charge is unmistakable), then sleeps again so the
        # charge sits strictly inside the wrapped life.
        dj_file = td / "disjoint.json"
        stub = ("import sys, time; sys.path.insert(0, %r); time.sleep(0.7); "
                "from experiments.cpu_budget import charge_cpu_child; "
                "charge_cpu_child('t0_34_stubspec', 5.0); time.sleep(0.3)"
                % str(REPO))
        w0 = time.time()
        dj_rc = subprocess.run(
            _wrap_cmd("t0_34_dj", sys.executable, "-c", stub),
            cwd=REPO, env=env_for(dj_file),
            capture_output=True, timeout=60).returncode
        dj_wall = time.time() - w0
        dj_day_total = sum(b8["used_s"] for b8 in
                           CpuBudget(dj_file).data["days"].values()) \
            if dj_file.exists() else 0.0
        dj_stub_billed = _label_total(dj_file, "t0_34_stubspec")
        disjoint_ok = (dj_rc == 0
                       and dj_stub_billed == 0.0
                       and dj_day_total >= 0.9              # billing alive
                       and dj_day_total <= dj_wall + TEST_HB_S + SLACK_S)

        # default heartbeat is what the docstring promises
        default_hb_ok = HEARTBEAT_S == 600.0

    return {
        "detached_receipts_ok": float(live_receipt and sigkill_bounded),
        "live_load": round(live_load, 2),
        "straddle_ok": straddle_ok,
        "multi_day_ok": multi_day_ok,
        "conservation_ok": conservation_ok,
        "zero_len_ok": zero_len_ok,
        "fresh_admits": fresh_admits,
        "exhausted_refused": exhausted_refused,
        "load_refused": load_refused,
        "rc_propagates": rc_propagates,
        "alive_billing": alive_billing,
        "hb_total_s": round(hb_total, 3),
        "hb_covers_life": hb_covers_life,
        "billed_at_kill_s": round(billed, 3),
        "elapsed_at_kill_s": round(elapsed_at_kill, 3),
        "sigkill_bounded": sigkill_bounded,
        "payload_died": payload_died,
        "wiring_ok": wiring_ok,
        "launcher_ok": launcher_ok,
        "e2e_total_s": round(e2e_total, 3),
        "live_receipt": live_receipt,
        "dj_wall_s": round(dj_wall, 3),
        "dj_day_total_s": round(dj_day_total, 3),
        "dj_stub_billed_s": round(dj_stub_billed, 3),
        "disjoint_ok": disjoint_ok,
        "default_hb_ok": default_hb_ok,
    }


def _control(seed: int) -> dict:
    """Two accountants that must FAIL, each alive while failing.

    The lump-sum accountant bills the whole interval at exit, to the exit
    day: it must FAIL day-split specifically — yesterday's bucket empty, the
    exit day holding everything — while the TOTAL stays exact.

    The double-billing accountant is the pre-fix composition, reproduced
    through the real path: the same wrap-a-descendant run as property 8, but
    the descendant strips the ownership marker before debiting — so its
    charge LANDS (alive) alongside the wrapper's heartbeat, and the day
    total overshoots the tree's true wall clock past the disjointness bound
    (fails property 8's arithmetic)."""
    base = time.mktime((2026, 9, 1, 0, 0, 0, 0, 0, -1))
    t0, t1 = base - 60.0, base + 120.0
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lump.json"
        exit_day = time.strftime("%Y-%m-%d", time.localtime(t1))
        CpuBudget(path).charge("t0_34_lump", t1 - t0, day=exit_day)
        lb = CpuBudget(path)
        yesterday, today = lb.used_s("2026-08-31"), lb.used_s("2026-09-01")

        dbl_file = Path(td) / "double.json"
        env = dict(os.environ)
        env[ENV_OVERRIDE] = str(dbl_file)
        env[ENV_HEARTBEAT] = str(TEST_HB_S)
        env.pop("JACK_AWAITING_SPEC", None)
        env.pop(ENV_WRAPPED, None)
        stub = ("import os, sys, time; sys.path.insert(0, %r); "
                "time.sleep(0.7); os.environ.pop(%r, None); "
                "from experiments.cpu_budget import charge_cpu_child; "
                "charge_cpu_child('t0_34_ctl_stub', 5.0); time.sleep(0.3)"
                % (str(REPO), ENV_WRAPPED))
        w0 = time.time()
        subprocess.run(_wrap_cmd("t0_34_ctl", sys.executable, "-c", stub),
                       cwd=REPO, env=env, capture_output=True, timeout=60)
        wall = time.time() - w0
        total = sum(bk["used_s"] for bk in
                    CpuBudget(dbl_file).data["days"].values()) \
            if dbl_file.exists() else 0.0
        stub_billed = _label_total(dbl_file, "t0_34_ctl_stub")
    return {
        "alive": today == 180.0,                      # the total landed…
        "lump_misbills": yesterday == 0.0 and today == 180.0,  # …in ONE day
        "dj_alive": stub_billed == 5.0,   # the stripped descendant's debit landed
        "dj_double_bills": total > wall + TEST_HB_S + SLACK_S,
    }


def _check(m: dict, c: dict):
    if m["live_load"] > LOAD_CEILING:
        return Status.VOID   # co-tenant condition; timing asserts unownable
    return (m["straddle_ok"] is True
            and m["multi_day_ok"] is True
            and m["conservation_ok"] is True
            and m["zero_len_ok"] is True
            and m["fresh_admits"] is True
            and m["exhausted_refused"] is True
            and m["load_refused"] is True
            and m["rc_propagates"] is True
            and m["alive_billing"] is True
            and m["hb_covers_life"] is True
            and m["sigkill_bounded"] is True
            and m["payload_died"] is True
            and m["wiring_ok"] is True
            and m["launcher_ok"] is True
            and m["live_receipt"] is True
            and m["disjoint_ok"] is True
            and m["default_hb_ok"] is True
            and c["alive"] is True
            and c["lump_misbills"] is True
            and c["dj_alive"] is True
            and c["dj_double_bills"] is True)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.34"], _experiment, _check, ledger=ledger,
                    control_fn=_control)
