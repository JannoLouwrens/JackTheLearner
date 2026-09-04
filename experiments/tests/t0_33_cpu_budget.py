"""T0.33 — CPU-hours on a shared box are accounted like GPU-hours.

This box hosts paying tenants. The GPU quota has had an accountant since
T0.12; the box's own CPU had none — the null baseline is exact: only GPU
hours are tracked, and the loop checks instantaneous load ONCE, at iteration
start (`ladder_loop.sh MAX_LOAD`), so a day of back-to-back cpu<2h children
was invisible to every instrument while it happened. The claim: every CPU
child the runner spawns debits a daily wall-clock budget
(`experiments/cpu_budget.json`), and the runner REFUSES to start a child when
the day's accumulated share or the box's current load would harm the tenants.

Properties, each independently checkable (T0.12's rewritten form is the
template — every accounting assertion is made at a MID-RANGE value that
moves, never at an exhausted one, where the assertion holds under every
implementation including a broken one):

  1. A charge MOVES the reading and PERSISTS across instances — the
     falsified_by's second clause ("a budget that reads the same whether or
     not runs happened") verbatim.
  2. Day boundaries isolate, asserted at mid-range: yesterday's charge does
     not move today's remaining.
  3. Affordability: a run whose worst-case child exceeds the remaining day is
     REFUSED, not attempted; a small one is admitted.
  4. The gate binds the right lane: cpu children gated and charged; GPU
     children unbound (their wall is remote waiting, metered by T0.12);
     cpu<48h refused with a ROUTING reason (the detached lane, which this
     meter honestly does not see).
  5. NO FORECLOSURE: the largest legal child (cpu<2h x 3 seeds x 2 = 54000 s)
     is admitted from a fresh day, and no registered cpu spec's canonical
     estimate exceeds the full ceiling — a tenant gate must not silently
     foreclose a cost class by arithmetic (the ME.11.E disease). Honest
     limit (68th audit B3): this gate compares against the FRESH ceiling, so
     it is vacuously green on any used day and can never fire while the day
     fills — and a refusal returns UNRECORDED by design, so a foreclosed day
     produces no number anywhere. The repair is a METRIC, not a gate:
     `n_foreclosed_now` counts the registered runner-lane cpu specs whose
     canonical estimate exceeds the LIVE day's remaining seconds at
     certificate time. It is reported so a foreclosed day stops being
     invisible; it is NOT gated at zero, because a legitimately-spent day
     SHOULD refuse things.
  6. Load refuses independently of the budget, and the threshold equals
     `ladder_loop.sh`'s MAX_LOAD — one number, two languages, parsed not
     assumed.
  7. An overrun past the ceiling leaves a mark (the meter observes what the
     refusal could not prevent — T0.12 property 7).
  8. A stale writer cannot erase another process's charges: two instances
     loaded from the same file both charge; the file shows both (the
     2026-08-12 GPU clobber, one resource over).
  9. WIRING, scanned live on `run._run_isolated`'s source: the gate is called
     BEFORE the child spawns, the debit after (on the success AND the timeout
     path — a killed child occupied the box for its full window), and a
     refusal returns UNRECORDED (an ERROR row would supersede a real result
     with scheduling noise).
 10. SHIPPED-PATH refusal, end to end: `_run_isolated` itself, pointed at an
     exhausted temp budget via JACK_CPU_BUDGET, refuses a real cheap spec
     before spawning — no child, no ledger byte moves. (If the refusal is
     broken this deliberately runs T0.01 for real, a few seconds and an
     honest re-run recorded by the runner — disclosed, not hidden.)
 11. THE RECEIPT IS COMMITTABLE (added 2026-09-04): the file this accountant
     writes is in `protocol.RUNNER_OUTPUTS` (with its `.tmp` staging sibling)
     and in `ladder_loop.sh`'s HARVEST_PATHS, both parsed live. Without the
     first, every child's receipt `+dirty`-stamps the NEXT certificate run in
     the same uncommitted window — the 2026-08-12 gpu_budget defect re-paid
     one file over; T0.34's attempt-1 cert (`dd8a2dd`) was the scar. Without
     the second, a pace-skip harvest commits a detached row while orphaning
     the CPU receipt accounting for it (29th audit B4's class).
 12. THE PROJECTION ONLY TIGHTENS (added 2026-09-04, 69th audit B4). The
     admission estimate is now `child_estimate_s` — the spec's own MEASURED
     last child duration where the ledger has one — and over EVERY registered
     runner-lane cpu spec it must sit at or below `spec_child_timeout_seconds`,
     the enum worst case it replaced. That is the safety argument stated as a
     checkable property rather than asserted in a docstring: admission after
     B4 is never stricter than before it, and the hard child-kill timeout is
     untouched, so an admitted child's true worst case is unchanged.
 13. THE PROJECTION BINDS, asserted at a MID-RANGE value (the T0.12 template).
     For a witness whose measurement is well under its enum, on a day drained
     to exactly halfway between the two, the ENUM would refuse and the
     projection ADMITS. Without this, property 12 is satisfied by an estimator
     that changed nothing.
 14. THE FALLBACK IS THE ENUM, ASSERTED BY DELETION. Pointed at an EMPTY
     ledger, every runner-lane spec's estimate equals its enum and says so in
     its provenance — the "no projection exists" branch exercised by removing
     the evidence, not by trusting that the code path is reachable.
 15. A CORRUPT LEDGER FAILS CLOSED. Pointed at unparseable bytes the estimate
     is the enum, not zero and not an exception. A projection that failed OPEN
     would admit everything on the day its input broke — the failure mode
     T0.12's "+inf load" rule exists for, pointed the other way because here
     the conservative side is the LARGER number.
 16. THE SLACK LINE PREDICTS THE REFUSAL (added 2026-09-04, 70th audit B4).
     `run status` prints, per cost class, `slack_s = CPU_DAY_CEILING_S −
     max(child_estimate_s over the class's live population)` — the day's
     spend at which the class starts foreclosing — because the bare count it
     used to print has no floor and no denominator and GROWS with the
     registry (three specs registered on 09-04 moved it 36 → 39 and nothing
     went amber). A printed number that does not bind is decoration, so the
     equivalence `used_s > slack_s` <=> `n_foreclosed >= 1` is asserted, at a
     MID-RANGE value: one second either side of the tightest class's own
     slack on a temp day, where the wrong side of the comparison flips the
     answer. Plus, on the LIVE day, that the two readers of the same
     arithmetic add up — `sum(n_foreclosed)` equals `n_foreclosed_now` and
     `sum(n)` equals this file's INDEPENDENT count of the gated population,
     so a class row can never disagree with the refusal it predicts.
     Instrumentation only: the ceiling is D20's and the owner's and nothing
     here moves it.

What B4 did NOT fix, recorded here because the metric below is the only place
it shows: the residual foreclosure is entirely specs with NO recorded duration
— `CPU_DAY_CEILING_S` (57600 s) is only 1.067x the largest legal child
(54000 s), so any never-run `cpu<2h` spec is refused past 6.25% of a day
whatever this projection does. Measured at the certificate: 53 foreclosed on
the enum, 36 on the projection, and all 36 are unmeasured specs. That is a
CEILING question (raising it loosens a tenant protection), routed to
`cpu48h-class-self-forecloses-the-day-meter` rather than fixed here.

Control (registry: "A leaky accountant must FAIL isolation, and the assertion
must be made at a MID-RANGE value"): `_LeakyBudget` collapses every day into
one bucket — the same disease as the retired ISO GPU week format, which
charged Sunday's runs to the exhausted week. Its isolation property must FAIL
(yesterday's charge visibly moves today's reading) while its arithmetic stays
alive (the charge itself lands), so the control cannot pass vacuously by
being broken everywhere.

Properties 12-15 carry their control INLINE rather than in `_control`, because
the estimator B4 replaced is still callable: property 13 refuses through the
pre-B4 path (`ledger_path` pointed at an empty ledger — the enum branch) on the
exact day the new path admits, and property 14 reconstructs the pre-B4 gate for
the WHOLE registry by deleting the evidence. A control that only failed
isolation would say nothing about any of them.

VOID lane: a live 1-minute load above LOAD_CEILING is a co-tenant condition —
the shipped-path refusal would fire for load rather than budget and the
admission properties cannot be honestly asserted. Instrument fault, not
refutation.

The STILL OPEN paragraph that stood here is CLOSED by T0.34 (2026-09-04):
the detached lane (`launch_detached.sh`, cpu<48h) now admits against the
same day ledger and bills per heartbeat — this spec's cpu<48h property
still checks only the ROUTING (the runner must not pretend to meter a lane
it never sees); the routed-to lane's receipts are T0.34's claim. Modules
invoked BY HAND remain unmetered: a human at a shell is the owner's lane.

Uses temp budget files; must never touch the real accounting.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import tempfile
import time
from pathlib import Path

from ..cpu_budget import (BUDGET_FILE, CPU_DAY_CEILING_S, ENV_OVERRIDE,
                          LOAD_CEILING, CpuBudget, _loadavg, child_estimate_s,
                          class_slack, foreclosed_now, gate_cpu_child,
                          measured_child_seconds)
from ..protocol import RUNNER_OUTPUTS, Ledger, Status, run_spec
from ..registry import BY_ID
from ..rtf import spec_child_timeout_seconds

REPO = Path(__file__).resolve().parents[2]

# Every property here is a property of experiments/cpu_budget.py or of the
# two files property 6/11 parse it against; run.py's wiring is scanned LIVE at
# run time (property 9), so a drift there is caught by the next run rather
# than by a stale flag. protocol.py and ladder_loop.sh joined 2026-09-04 with
# property 11: this test asserts parsed equality with both, so an edit to
# either must stale this certificate rather than decay it silently.
IMPL_DEPS = ["experiments/cpu_budget.py", "experiments/protocol.py",
             "scripts/ladder_loop.sh"]

MID_CHARGE_S = 600.0          # the mid-range value the isolation assert uses
WORST_LEGAL_CHILD_S = 54000.0  # cpu<2h x 3 seeds x 2 — must fit a fresh day
GPU_WITNESS = "T2.01"          # gpu<8h — must be unbound by this meter
DETACHED_WITNESS = "PS.04"     # cpu<48h — must be refused with a routing reason
SHIPPED_WITNESS = "T0.01"      # cheap real spec offered to the shipped path
WORST_WITNESS = "XL.01"        # cpu<2h, 3 seeds: the largest legal child
PROBE_DURATION_S = 100.0       # synthetic measurement for property 13


def _runner_cpu_specs() -> list:
    """The lane this meter gates: cpu specs the runner spawns as children.
    cpu<48h is the detached lane (property 4's routing reason)."""
    return [s for s in BY_ID.values()
            if s.budget.value.startswith("cpu")
            and s.budget.value != "cpu<48h"]


def _yesterday() -> str:
    return time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))


class _LeakyBudget(CpuBudget):
    """The control: every day is one bucket, so isolation MUST fail."""

    def _key(self, day: str | None = None) -> str:
        return "ALL-DAYS"


def _experiment(seed: int) -> dict:
    live_load = _loadavg()
    day0 = time.strftime("%Y-%m-%d")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cpu_budget.json"
        b = CpuBudget(path)

        starts_full = b.remaining_s(day0) == CPU_DAY_CEILING_S
        b.charge("T0.33-probe", MID_CHARGE_S, day=day0)
        charged_moves = b.used_s(day0) == MID_CHARGE_S
        persists = CpuBudget(path).used_s(day0) == MID_CHARGE_S

        # Isolation at MID-RANGE: yesterday's charge, today's exact reading.
        b.charge("T0.33-probe", 7200.0, day=_yesterday())
        remaining_mid = CpuBudget(path).remaining_s(day0)
        isolation_ok = remaining_mid == CPU_DAY_CEILING_S - MID_CHARGE_S

        affords_small = b.afford(MID_CHARGE_S, day=day0)
        refuses_oversized = not b.afford(remaining_mid + 1.0, day=day0)

        # Stale writer: both instances loaded now; both charges must survive.
        b1, b2 = CpuBudget(path), CpuBudget(path)
        b1.charge("T0.33-w1", 100.0, day=day0)
        b2.charge("T0.33-w2", 200.0, day=day0)
        stale_writer_safe = (CpuBudget(path).used_s(day0)
                             == MID_CHARGE_S + 300.0)

        # Gates, load injected so a busy co-tenant cannot decide them.
        worst = BY_ID[WORST_WITNESS]
        fresh_path = Path(td) / "fresh.json"
        # NO FORECLOSURE is a property of the WORST CASE, so it is asserted
        # against the empty ledger — the enum branch — not against whatever
        # XL.01 happens to have measured. B4's projection may only make this
        # easier; it must not be what makes it true.
        empty_ledger = Path(td) / "empty_ledger.json"
        empty_ledger.write_text(json.dumps({"results": {}}))
        d = gate_cpu_child(worst, path=fresh_path, loadavg=0.0,
                           ledger_path=empty_ledger)
        fresh_admits_worst = d.admitted and d.est_s == WORST_LEGAL_CHILD_S

        CpuBudget(fresh_path).charge("T0.33-drain", CPU_DAY_CEILING_S)
        d = gate_cpu_child(worst, path=fresh_path, loadavg=0.0)
        exhausted_refused = (not d.admitted) and "day budget" in d.reason

        d = gate_cpu_child(worst, path=Path(td) / "fresh2.json",
                           loadavg=LOAD_CEILING + 1.0)
        load_refused = (not d.admitted) and "load" in d.reason

        d = gate_cpu_child(BY_ID[GPU_WITNESS], path=fresh_path, loadavg=0.0)
        gpu_unbound = d.admitted and "T0.12" in d.reason

        d = gate_cpu_child(BY_ID[DETACHED_WITNESS], path=fresh_path,
                           loadavg=0.0)
        detached_routed = (not d.admitted) and "detached" in d.reason

        # Overrun leaves a mark.
        over_path = Path(td) / "over.json"
        ob = CpuBudget(over_path)
        ob.charge("T0.33-over", CPU_DAY_CEILING_S + 1.0)
        marks = CpuBudget(over_path).data["overruns"]
        overrun_marked = len(marks) == 1 and marks[0]["spec_id"] == "T0.33-over"

        # No registered cpu spec is foreclosed by the ceiling's arithmetic.
        foreclosed = sorted(
            s.id for s in _runner_cpu_specs()
            if spec_child_timeout_seconds(s) > CPU_DAY_CEILING_S)

        # METRIC, not gate (68th audit B3): how many runner-lane cpu specs
        # the LIVE day would refuse right now. Reads the real accounting
        # (read-only — "never touch" bans writes, not sight); a used day
        # refusing work is the protection working, but it must be a number
        # somewhere, because the refused work never runs and never reports.
        live_remaining = CpuBudget().remaining_s()
        n_foreclosed_now = len(foreclosed_now())
        # The same count on the estimator B4 replaced, so the certificate
        # carries what the change actually bought on the day it landed
        # rather than a claim about it.
        n_foreclosed_enum = sum(
            1 for s in _runner_cpu_specs()
            if spec_child_timeout_seconds(s) > live_remaining)
        # …and the residual, named: how many of the still-foreclosed have no
        # measurement to project from. B4 cannot help those, and if this ever
        # falls below n_foreclosed_now the projection is failing on specs it
        # DOES have evidence for.
        fore_ids = set(foreclosed_now())
        n_foreclosed_unmeasured = sum(
            1 for s in _runner_cpu_specs()
            if s.id in fore_ids and measured_child_seconds(s.id) is None)

        # ── 12/13/14/15: the admission estimate ──────────────────────────
        # 12. ONLY TIGHTENS, over the live registry against the real ledger.
        est_above_enum = sorted(
            s.id for s in _runner_cpu_specs()
            if child_estimate_s(s)[0] > float(spec_child_timeout_seconds(s)))
        projection_only_tightens = est_above_enum == []

        # 14/15. The two fallback branches, exercised by DELETION and by
        # CORRUPTION — never by trusting that the branch is reachable.
        corrupt_ledger = Path(td) / "corrupt_ledger.json"
        corrupt_ledger.write_bytes(b"{not json at all")
        enum_only, enum_prov, corrupt_ok = True, True, True
        for s in _runner_cpu_specs():
            enum_s = float(spec_child_timeout_seconds(s))
            e_est, e_prov = child_estimate_s(s, empty_ledger)
            c_est, _c_prov = child_estimate_s(s, corrupt_ledger)
            enum_only &= (e_est == enum_s)
            enum_prov &= e_prov.startswith("ENUM")
            corrupt_ok &= (c_est == enum_s)
        empty_ledger_falls_back = bool(enum_only and enum_prov)
        corrupt_ledger_fails_closed = bool(corrupt_ok)

        # 13. THE PROJECTION BINDS, at a MID-RANGE value. A synthetic
        # measurement, so the assertion is about the estimator and cannot
        # flap on whatever XL.01 last happened to cost.
        proj_ledger = Path(td) / "proj_ledger.json"
        proj_ledger.write_text(json.dumps(
            {"results": {WORST_WITNESS: {"duration_s": PROBE_DURATION_S}}}))
        enum_worst = float(spec_child_timeout_seconds(worst))
        proj_worst, prov_worst = child_estimate_s(worst, proj_ledger)
        mid_path = Path(td) / "mid.json"
        mid_remaining = (proj_worst + enum_worst) / 2.0
        CpuBudget(mid_path).charge("T0.33-mid",
                                   CPU_DAY_CEILING_S - mid_remaining)
        d_proj = gate_cpu_child(worst, path=mid_path, loadavg=0.0,
                                ledger_path=proj_ledger)
        d_enum = gate_cpu_child(worst, path=mid_path, loadavg=0.0,
                                ledger_path=empty_ledger)
        projection_binds = (proj_worst < enum_worst
                            and prov_worst.startswith("MEASURED")
                            and d_proj.admitted
                            and not d_enum.admitted
                            and "day budget" in d_enum.reason)

        # ── 16: THE SLACK LINE PREDICTS THE REFUSAL ──────────────────────
        # `run status` now prints, per cost class, the spend at which the
        # class starts foreclosing. A printed number that does not bind is
        # decoration, so the equivalence is asserted rather than argued —
        # and at a MID-RANGE value (the T0.12 template): one second either
        # side of the tightest class's own slack, on a temp day, where a
        # wrong side of the comparison flips the answer.
        # `fresh_path` was drained by property 3 — a never-written path is the
        # only honest zero here.
        zero_path = Path(td) / "slack_zero.json"
        tight = class_slack(zero_path, empty_ledger)[0]
        under = Path(td) / "slack_under.json"
        CpuBudget(under).charge("T0.33-slack", tight["slack_s"] - 1.0)
        over = Path(td) / "slack_over.json"
        CpuBudget(over).charge("T0.33-slack", tight["slack_s"] + 1.0)

        def _row(p, b):
            return next(r for r in class_slack(p, empty_ledger)
                        if r["budget"] == b)

        r_under, r_over = _row(under, tight["budget"]), _row(over, tight["budget"])
        # ...and the live day, every class at once: the printed state and the
        # gate's own answer are one comparison, so they may never disagree.
        live_rows = class_slack()
        live_agrees = all(
            (r["used_s"] > r["slack_s"]) == (r["n_foreclosed"] >= 1)
            for r in live_rows)
        slack_predicts_refusal = (
            tight["slack_s"] > 0.0                    # not foreclosed fresh
            and r_under["n_foreclosed"] == 0          # one second under: open
            and r_over["n_foreclosed"] >= 1           # one second over: closing
            and len(foreclosed_now(under, empty_ledger)) == 0
            and len(foreclosed_now(over, empty_ledger)) == r_over["n_foreclosed"]
            and live_agrees
            # the two readers of the same arithmetic add up
            and sum(r["n_foreclosed"] for r in live_rows) == n_foreclosed_now
            and sum(r["n"] for r in live_rows) == len(_runner_cpu_specs()))

        # One threshold, two languages.
        loop_src = (REPO / "scripts" / "ladder_loop.sh").read_text()
        m = re.search(r"^MAX_LOAD=([0-9.]+)", loop_src, re.M)
        loop_load_agrees = m is not None and float(m.group(1)) == LOAD_CEILING

        # One receipt, three registers: the file this accountant writes must
        # be in RUNNER_OUTPUTS (else every child's receipt `+dirty`-stamps the
        # NEXT certificate — T0.34 attempt 1, `dd8a2dd`) and in the loop's
        # harvest pathspec (else a pace-skip harvest commits a ledger row
        # while orphaning the receipt that accounts for it — 29th audit B4,
        # one lane over). Parsed from the live constant and the live script,
        # never assumed; red on the tree of 2026-09-03.
        rel = BUDGET_FILE.relative_to(REPO).as_posix()
        in_outputs = (rel in RUNNER_OUTPUTS
                      and rel + ".tmp" in RUNNER_OUTPUTS)
        hp = re.search(r'^HARVEST_PATHS="([^"]*)"', loop_src, re.M)
        receipt_committable = (in_outputs and hp is not None
                               and rel in hp.group(1).split())

        # Wiring, scanned on the live function.
        from .. import run as run_mod
        src = inspect.getsource(run_mod._run_isolated)
        spawn_at = src.find("sp.run([sys.executable")
        gate_at = src.find("gate_cpu_child(_spec)")
        bills = [i for i in range(len(src)) if src.startswith("_bill_cpu(_t0)", i)]
        wiring_ok = (0 <= gate_at < spawn_at
                     and "charge_cpu_child(spec_id" in src
                     and len(bills) == 2          # timeout path AND success path
                     and all(i > spawn_at for i in bills)
                     and "REFUSED before start" in src
                     and "ledger.record" not in src[gate_at:spawn_at])

        # Shipped path: the real runner refuses a real spec on an exhausted
        # day, before spawning, without a ledger byte moving.
        exhausted = Path(td) / "exhausted.json"
        CpuBudget(exhausted).charge("T0.33-drain", CPU_DAY_CEILING_S)
        ledger_path = REPO / "experiments" / "ledger.json"
        before = ledger_path.read_bytes()
        old_env = os.environ.get(ENV_OVERRIDE)
        os.environ[ENV_OVERRIDE] = str(exhausted)
        try:
            t0 = time.perf_counter()
            res = run_mod._run_isolated(SHIPPED_WITNESS, Ledger())
            shipped_s = time.perf_counter() - t0
        finally:
            if old_env is None:
                os.environ.pop(ENV_OVERRIDE, None)
            else:
                os.environ[ENV_OVERRIDE] = old_env
        shipped_refusal = (res.status is Status.ERROR
                           and "REFUSED before start" in res.message
                           and "day budget" in res.message
                           and ledger_path.read_bytes() == before
                           and shipped_s < 2.0)   # refused, not run-and-failed

    return {
        "cpu_quota_enforced": float(shipped_refusal and exhausted_refused),
        "live_load": round(live_load, 2),
        "starts_full": starts_full,
        "charged_moves": charged_moves,
        "persists": persists,
        "remaining_mid_s": remaining_mid,
        "isolation_ok": isolation_ok,
        "affords_small": affords_small,
        "refuses_oversized": refuses_oversized,
        "stale_writer_safe": stale_writer_safe,
        "fresh_admits_worst": fresh_admits_worst,
        "exhausted_refused": exhausted_refused,
        "load_refused": load_refused,
        "gpu_unbound": gpu_unbound,
        "detached_routed": detached_routed,
        "overrun_marked": overrun_marked,
        "cpu_foreclosed": foreclosed,
        "live_remaining_s": round(live_remaining, 2),
        "n_foreclosed_now": n_foreclosed_now,
        "n_foreclosed_enum": n_foreclosed_enum,
        "n_foreclosed_unmeasured": n_foreclosed_unmeasured,
        "slack_predicts_refusal": slack_predicts_refusal,
        "class_slack": [{k: r[k] for k in ("budget", "n", "slack_s",
                                           "used_s", "n_foreclosed")}
                        for r in live_rows],
        "projection_only_tightens": projection_only_tightens,
        "est_above_enum": est_above_enum,
        "projection_binds": projection_binds,
        "empty_ledger_falls_back": empty_ledger_falls_back,
        "corrupt_ledger_fails_closed": corrupt_ledger_fails_closed,
        "loop_load_agrees": loop_load_agrees,
        "receipt_committable": receipt_committable,
        "wiring_ok": wiring_ok,
        "shipped_refusal": shipped_refusal,
        "shipped_refusal_s": round(shipped_s, 3),
    }


def _control(seed: int) -> dict:
    """The leaky accountant. It must FAIL isolation specifically — and be
    demonstrably alive while failing, so a broken-everywhere control cannot
    pass by never charging at all."""
    day0 = time.strftime("%Y-%m-%d")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "leaky.json"
        lb = _LeakyBudget(path)
        lb.charge("T0.33-leak", MID_CHARGE_S, day=_yesterday())
        today_reading = _LeakyBudget(path).used_s(day0)
    return {
        "alive": today_reading > 0.0,                 # the charge landed
        "leak_visible": today_reading == MID_CHARGE_S,  # ...in TODAY's bucket
    }


def _check(m: dict, c: dict):
    # Co-tenant load spike: admissions cannot be asserted and the shipped
    # refusal cannot be attributed to the budget. Instrument fault.
    if m["live_load"] > LOAD_CEILING:
        return Status.VOID
    return (m["starts_full"] is True
            and m["charged_moves"] is True
            and m["persists"] is True
            and m["isolation_ok"] is True
            and m["remaining_mid_s"] == CPU_DAY_CEILING_S - MID_CHARGE_S
            and m["affords_small"] is True
            and m["refuses_oversized"] is True
            and m["stale_writer_safe"] is True
            and m["fresh_admits_worst"] is True
            and m["exhausted_refused"] is True
            and m["load_refused"] is True
            and m["gpu_unbound"] is True
            and m["detached_routed"] is True
            and m["overrun_marked"] is True
            and m["cpu_foreclosed"] == []
            and m["projection_only_tightens"] is True
            and m["est_above_enum"] == []
            and m["projection_binds"] is True
            and m["empty_ledger_falls_back"] is True
            and m["corrupt_ledger_fails_closed"] is True
            and m["slack_predicts_refusal"] is True
            and m["loop_load_agrees"] is True
            and m["receipt_committable"] is True
            and m["wiring_ok"] is True
            and m["shipped_refusal"] is True
            and c["alive"] is True
            and c["leak_visible"] is True)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.33"], _experiment, _check, ledger=ledger,
                    control_fn=_control)
