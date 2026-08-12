"""T0.26 — a rig-health gate must refuse a broken world and admit an honest one.

A pre-registered threshold is a claim that the gated statistic's attainable
range under THIS rig straddles it. BA.01 broke that claim in both directions
in one day: v2 redefined the statistic so the rig's own uniform hold cleared
the unmoved gate 4.7x (INERT — a zero-fall-variance world reads healthy), and
v3 restored the statistic to find its 2.5 unreachable on open ground (a TAIL
LOTTERY — honest worlds VOID unless they draw structure outliers). Law 4
watches the number; nothing watched the measurement. This spec does, executably.

Both directions run through BA.01's OWN machinery — `rollout_rig` drives the
same `_episode` the recorded runs use, `rig_health` is the same statistic path
`_evaluate` gates on — never a restatement (the T0.16 lesson: a tidied copy
passes while the shipped computation drifts). The degenerate rig is DECLARED
IN BA.01, not invented here (the LC.01 lesson: the artifact names the object
under audit): one fixed 6.3-degree tilt, zero kick, zero arm noise, every
spawn at the model-derived most-open cell — failure mode #2 verbatim, every
fall on one schedule, while the hold keeps ABSOLUTE topple times wide.

PROPERTIES (world seed 90 — BA.01's pilot world, disjoint from its registered
seeds — 60 episodes per rig; pilot numbers beside each):

  P1 broken_world_refused   degenerate tf_fall_spread < TF_FALL_SPREAD_MIN
                            and rig_ok == 0 (pilot: 0.0 vs 2.5, refused)
  P2 degeneracy_isolated    the degenerate world passes every OTHER rig gate
                            (toppled_frac >= 0.60, tf_abs_spread >= 2.5;
                            pilot 1.0 and 11.13) — broken ONLY in the gated
                            dimension, else P1 could ride a different gate
  P3 honest_bulk_admitted   honest tf_fall_spread >= TF_FALL_SPREAD_MIN and
                            rig_ok == 1 (pilot: 9.38, admitted) — the
                            reachability direction v3 lacked

CONTROL: the pre-fix (v2) gate — toppled_frac + tf_abs_spread, no fall term —
replayed verbatim against the SAME degenerate episodes. It MUST certify the
broken world healthy. If it ever refuses, the fixture no longer reproduces
the v2 disease and this spec guards nothing.
"""
from __future__ import annotations

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from . import ba_01_feels_the_fall as ba

WORLD_SEED = 90     # BA.01's pilot world; its registered seeds are 0/1/2
N_EP = 60

_CACHE: dict = {}


def _rigs() -> dict | None:
    """Both rigs' health, measured once through BA.01's real episode path.
    None on a PS.01 borrow refusal (VOID, per BA.01's own convention)."""
    if "rigs" not in _CACHE:
        eps_d = ba.rollout_rig(WORLD_SEED, N_EP, degenerate=True)
        eps_h = ba.rollout_rig(WORLD_SEED, N_EP, degenerate=False)
        if eps_d is None or eps_h is None:
            _CACHE["rigs"] = None
        else:
            _CACHE["rigs"] = {"deg": ba.rig_health(eps_d),
                              "hon": ba.rig_health(eps_h)}
    return _CACHE["rigs"]


def _prefix_rig_ok(rig: dict) -> float:
    """THE PRE-FIX (v2) GATE, verbatim shape (ba_01 @ 0fce271): rig health
    was toppled_frac and ABSOLUTE spread alone — no fall-dynamics term, so
    the hold's own uniform t_r could certify a zero-fall-variance world.
    Kept executable as this spec's control."""
    return 1.0 if (rig["toppled_frac"] >= ba.TOPPLED_FRAC_MIN
                   and rig["tf_abs_spread"] >= ba.TF_ABS_SPREAD_MIN) else 0.0


def _experiment(seed: int) -> dict:
    rigs = _rigs()
    if rigs is None:
        return {"probe": "VOID", "properties_failed": float("nan")}
    d, h = rigs["deg"], rigs["hon"]
    p1 = (d["tf_fall_spread"] < ba.TF_FALL_SPREAD_MIN and d["rig_ok"] == 0.0)
    p2 = (d["toppled_frac"] >= ba.TOPPLED_FRAC_MIN
          and d["tf_abs_spread"] >= ba.TF_ABS_SPREAD_MIN)
    p3 = (h["tf_fall_spread"] >= ba.TF_FALL_SPREAD_MIN and h["rig_ok"] == 1.0)
    return {"p1_broken_world_refused": float(p1),
            "p2_degeneracy_isolated": float(p2),
            "p3_honest_bulk_admitted": float(p3),
            "properties_failed": float(3 - int(p1) - int(p2) - int(p3)),
            "deg_tf_fall_spread": d["tf_fall_spread"],
            "deg_tf_abs_spread": d["tf_abs_spread"],
            "deg_toppled_frac": d["toppled_frac"],
            "deg_rig_ok": d["rig_ok"],
            "hon_tf_fall_spread": h["tf_fall_spread"],
            "hon_tf_abs_spread": h["tf_abs_spread"],
            "hon_toppled_frac": h["toppled_frac"],
            "hon_rig_ok": h["rig_ok"]}


def _control(seed: int) -> dict:
    """The v2 gate against the same broken world. It must say 'healthy'."""
    rigs = _rigs()
    if rigs is None:
        return {"probe": "VOID",
                "prefix_certifies_broken_world": float("nan")}
    return {"prefix_certifies_broken_world": _prefix_rig_ok(rigs["deg"])}


def _check(m: dict, c: dict):
    if m.get("probe") == "VOID" or c.get("probe") == "VOID":
        return Status.VOID
    return (m["properties_failed"] == 0.0
            and c["prefix_certifies_broken_world"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.26"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    print(run().status)
