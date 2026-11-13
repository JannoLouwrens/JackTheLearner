"""w_7_declared_clock.py — W.7: time compression is a DECLARED transformation.

Binding form: the control re-registered 2026-11-13 (Director ruling 1, ORDERS
J6(a), commit 98134d7). W0 has ONE declared clock, k = 72 (one Jack-day =
1200 sim-s). Every rate-bearing channel's sim-time constant must equal its
SOURCED human constant / 72, asserted against the DECLARED k — never against
another channel and never against SIM_DAY_S (two quantities derived from the
same wrong k agree perfectly).

Each channel is MEASURED on the code the world runs, through the harnesses
that already certify it, not read off a constant:

  day       86400 / needs.SIM_DAY_S                 (the declaration itself)
  thermal   W.1's 1/e relaxation of the shipped thermal_step at 1 met,
            against W.1's sourced reference tau (70 kg x 3470 J/kg/K / hA)
  thirst    W.2's deprived statue, time to death, against 3 days
  hunger    W.2's water-met statue, time to death, against 21 days
  wake      W.2's fitted sleep-pressure tau_wake, against 18.2 h
  sleep     W.2's fitted tau_sleep, against 4.2 h

implied k = human constant / measured sim constant. A channel passes when
|k/72 - 1| <= K_TOL_FRAC. The tolerance is the registered one: "the bars' own:
W.1(b)'s 1% bar admits ~+/-11.6% in k" — taken as registered, not tuned.

CONTROLS (must behave as stated, or the gate measures nothing):
  * planted channels at k = 1 and k = 60, built as pure relaxations with
    tau = reference / k and pushed through the SAME measurement and the SAME
    judgement, must FAIL;
  * a planted channel at exactly k = 72 must PASS (a checker never seen
    passing is as untested as one never seen failing — LESSONS_CORE 9);
  * NULL (registry: "a compression test that cannot tell k = 1 from k = 72 is
    measuring nothing"): the shipped channel table judged against a declared
    k = 1 must FAIL.

Expected on the shipped world (W.2's 08-30 row): thermal passes (k ~71.1),
thirst (~455) and hunger (~1067 measured) fail. The sleep clocks were not in
the order's expectation: tau_wake 700 s implies k = 93.6 and tau_sleep 160 s
implies 94.5 — both outside +/-11.6%. That is a finding for the needs-clock
race (J6(b)), not a reason to drop the channel.
"""
from __future__ import annotations

import math

from .. import needs as N
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from . import w_1_heat_balance as W1
from . import w_2_needs_ledger as W2

SPEC_ID = "W.7"
IMPL_DEPS = ["playground.py", "experiments/needs.py", "experiments/drives.py"]

K_DECLARED = 72.0                 # ruling 1. A constant, never read off the world.
K_TOL_FRAC = 0.116                # registered: W.1(b)'s 1% bar ~ +/-11.6% in k
K_PLANTED_OFF = (1.0, 60.0)       # registered planted controls: must FAIL
K_NULL = 1.0                      # registry null

HUMAN_S = {                        # sourced human constants, in real seconds
    "day": 86400.0,
    "thermal": W1._tau_ref(0.0),   # 70 kg x 3470 J/kg/K / (h_r+h_c) A_DuBois
    "thirst": W2.HUMAN_THIRST_S,   # ~3 days
    "hunger": W2.HUMAN_HUNGER_S,   # 3 weeks
    "wake": W2.SLEEP_TAU_WAKE_H * 3600.0,    # 18.2 h
    "sleep": W2.SLEEP_TAU_SLEEP_H * 3600.0,  # 4.2 h
}
CHANNELS = tuple(HUMAN_S)


def _thermal_eq(step) -> float:
    return W1._settled_body_c(step, W1.SCENARIO_TENV_C, N.T_SETPOINT)


def _measure_thermal_tau(step) -> float:
    return W1._measured_tau(step, W1.SCENARIO_TENV_C, _thermal_eq(step))


def _planted_relaxation(tau_s: float):
    """A pure first-order channel with time constant tau_s about T_env + 7 C."""
    T_inf = W1.SCENARIO_TENV_C + 7.0

    def step(T: float, T_env: float, dt: float, occ: float = 0.0,
             wet: float = 0.0) -> float:
        return T + (T_inf - T) * dt / tau_s
    return step


def _judge(sim_s: dict, k_declared: float) -> dict:
    """The ONE judgement, used by the experiment, the controls and the null."""
    out, worst = {}, 0.0
    for ch, s in sim_s.items():
        k = HUMAN_S[ch] / s if s and math.isfinite(s) and s > 0 else float("nan")
        err = abs(k / k_declared - 1.0) if math.isfinite(k) else float("inf")
        out[f"k_{ch}"] = k
        out[f"err_{ch}"] = err
        out[f"ok_{ch}"] = float(err <= K_TOL_FRAC)
        worst = max(worst, err)
    out["compression_invariance_error"] = worst
    out["all_ok"] = float(all(out[f"ok_{ch}"] == 1.0 for ch in sim_s))
    return out


def _shipped_channels(seed: int) -> tuple[dict, dict]:
    rig = W2.Rig(seed)
    probe = rig.layer(seed)
    pre = dict(rig.preconditions(probe))
    dep = W2._drive(rig, rig.layer(seed), "none", W2.DEPRIVED_HORIZON_S)
    wat = W2._drive(rig, rig.layer(seed), "water", W2.DEPRIVED_HORIZON_S)
    sl = W2._sleep_clocks(rig, seed)
    sim = {
        "day": float(N.SIM_DAY_S),
        "thermal": _measure_thermal_tau(W1._make_ship_1met(0.0)),
        "thirst": dep["life_s"],
        "hunger": wat["life_s"],
        "wake": sl["d_tau_wake_s"],
        "sleep": sl["d_tau_sleep_s"],
    }
    diag = {"thirst_cause": dep["cause"], "hunger_cause": wat["cause"],
            "rig_thirst_is_dehydration": float(dep["cause"] == "dehydration"),
            "rig_hunger_is_starvation": float(wat["cause"] == "starvation"),
            "rig_sleep_fit_alive": float(sl["d_fit_died"] == 0.0)}
    for k in ("rig_land_sky_open", "rig_pool_mouth_near_surface",
              "rig_pool_head_not_submerged"):
        if k in pre:
            diag[k] = pre[k]
    return sim, diag


def _experiment(seed: int) -> dict:
    sim, diag = _shipped_channels(seed)
    m: dict = {f"sim_{ch}_s": v for ch, v in sim.items()}
    m.update({f"human_{ch}_s": HUMAN_S[ch] for ch in CHANNELS})
    m.update(_judge(sim, K_DECLARED))
    m["k_declared"] = K_DECLARED
    m["k_tol_frac"] = K_TOL_FRAC
    m["day_exact"] = float(86400.0 / N.SIM_DAY_S == K_DECLARED)
    # The registered NULL: the same table judged against k = 1 must fail.
    null = _judge(sim, K_NULL)
    m["null_k1_all_ok"] = null["all_ok"]
    m["null_k1_error"] = null["compression_invariance_error"]
    m["failing_channels"] = ",".join(
        ch for ch in CHANNELS if m[f"ok_{ch}"] != 1.0) or "none"
    m.update({k: v for k, v in diag.items() if isinstance(v, float)})
    m["thirst_cause"], m["hunger_cause"] = diag["thirst_cause"], diag["hunger_cause"]
    rig_ok = all(v == 1.0 for k, v in diag.items()
                 if k.startswith("rig_") and isinstance(v, float))
    m["rig_ok"] = float(rig_ok)
    if not rig_ok:
        m["void_reason"] = ("VOID — a scripted deprivation did not take "
                            "effect: " + ",".join(k for k, v in diag.items()
                                                  if k.startswith("rig_") and v != 1.0))
    return m


def _control(seed: int) -> dict:
    c: dict = {}
    ref = HUMAN_S["thermal"]
    for k in K_PLANTED_OFF + (K_DECLARED,):
        tau = _measure_thermal_tau(_planted_relaxation(ref / k))
        j = _judge({"thermal": tau}, K_DECLARED)
        tag = f"k{k:g}"
        c[f"ctl_{tag}_tau_s"] = tau
        c[f"ctl_{tag}_implied_k"] = j["k_thermal"]
        c[f"ctl_{tag}_ok"] = j["all_ok"]
    c["ctl_off_clock_all_fail"] = float(all(
        c[f"ctl_k{k:g}_ok"] == 0.0 for k in K_PLANTED_OFF))
    c["ctl_on_clock_passes"] = c[f"ctl_k{K_DECLARED:g}_ok"]
    return c


def _check(m: dict, c: dict) -> bool:
    """PRE-REGISTERED. Every measured channel on the declared clock, the day
    exactly 72, the k = 1 null rejected, both planted off-clock channels
    rejected and the planted on-clock channel accepted."""
    if m.get("rig_ok", 0.0) != 1.0:
        return False
    claim = m["all_ok"] == 1.0 and m["day_exact"] == 1.0
    null_ok = m["null_k1_all_ok"] == 0.0
    ctl_ok = (c["ctl_off_clock_all_fail"] == 1.0
              and c["ctl_on_clock_passes"] == 1.0)
    return bool(claim and null_ok and ctl_ok)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
