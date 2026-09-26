"""Branch battery for T2.11's `mi_beats_field` conjunct (t211 METRIC ruling).

Replays hand-built fold dicts through the REAL `_check` to prove each branch
fires on its named condition — including the one that matters most, that the
new conjunct CANNOT rescue the `beats_shuffled` red the spec is parked on.

Writes nothing, asserts loudly, costs milliseconds. Run:

    /data/venvs/jackthelearner/bin/python -m experiments.tests.t211_mi_battery

WHY IT IS A FILE AND NOT A PARAGRAPH. `mi_beats_field` was added to a spec
whose `run()` refuses (`_GATES_FROZEN` is False), so the conjunct's own
branches cannot be exercised by dispatching the spec — and a gate nobody has
seen fire is a gate on the author's word. `t301_shuffle_probe.py` is the
precedent: a sibling probe beside the spec it interrogates.
"""
from __future__ import annotations

from ..protocol import Status
from .t2_11_skills_distinguishable import (  # noqa: F401
    CHANCE, ABOVE_CHANCE_MIN, MARGIN_MIN, PER_CLASS_MIN, SHUFFLE_FIT_FLOOR,
    SHUFFLE_BAND, FLOOR_COVERAGE, ORACLE_MIN, MI_MARGIN_MIN, _check)


def base_m() -> dict:
    """A fold that clears every rig gate and all five claim gates."""
    return {
        "claim_acc": 0.90, "claim_per_class_min": 0.60,
        "margin_vs_shuffled": 0.30, "margin_vs_zero": 0.70,
        "mi_margin": 1.05,
        "shuffle_clf_fit": 0.90, "shuffle_clf_heldout": 0.10,
        "hash_overlap_max": 0, "min_coverage": 0.06, "zero_coverage": 0.10,
        "oracle_acc": 0.95, "zero_q_absmax": 0.0,
    }


def base_c() -> dict:
    """A control fold that fails the three shared claim gates."""
    return {"claim_acc": 0.20, "claim_per_class_min": 0.05,
            "margin_vs_zero": 0.05}


def show(name: str, m: dict, c: dict, expect) -> None:
    got = _check(m, c)
    ok = (got == expect) if isinstance(expect, Status) else (bool(got) is expect)
    print(f"{'ok  ' if ok else '!! MISFIRE'}  {name}: _check -> {got} "
          f"(expected {expect})")
    assert ok, name


def main() -> None:
    # 1. all green -> True. Without this the rest proves only that _check
    #    can say no.
    show("all-green", base_m(), base_c(), True)

    # 2. THE NEW BRANCH, AS THE SOLE CAUSE. Everything else green, mi_margin
    #    one hundredth under the bar -> False. This is the conjunct being
    #    load-bearing rather than decorative (T0.13's question).
    m = base_m(); m["mi_margin"] = MI_MARGIN_MIN - 0.01
    show("mi_margin just under bar -> False (sole cause)", m, base_c(), False)

    # 3. Exactly at the bar -> True. `>=` semantics, like every gate here.
    m = base_m(); m["mi_margin"] = MI_MARGIN_MIN
    show("mi_margin == bar -> True", m, base_c(), True)

    # 4. THE CANNOT-RESCUE REPLAY, and it is the reason this file exists.
    #    v3's measured worst-seed shape: the accuracy channel is RED
    #    (control beats claim by 0.1172) while MI separates by +0.6268. The
    #    new conjunct must NOT turn that into a PASS — a strengthening that
    #    can rescue an existing red is a weakening wearing a hat.
    m = base_m(); m["margin_vs_shuffled"] = -0.1172; m["mi_margin"] = 0.6268
    show("v3 worst-seed shape: acc red, MI green -> still False",
         m, base_c(), False)

    # 5. The mirror of 4: MI red, accuracy green -> also False. Neither
    #    channel may carry the verdict alone.
    m = base_m(); m["margin_vs_shuffled"] = 0.30; m["mi_margin"] = -0.01
    show("acc green, MI red -> still False", m, base_c(), False)

    # 6. Rig red (dead oracle) -> VOID regardless of mi_margin. An apparatus
    #    outcome must not become a refutation that fires `kills`.
    m = base_m(); m["oracle_acc"] = 0.10; m["mi_margin"] = 2.0
    show("rig red -> VOID even with MI at 2.0 nats", m, base_c(), Status.VOID)

    # 7. Control passes the shared gates -> False even with all else green.
    c = {"claim_acc": 0.90, "claim_per_class_min": 0.60,
         "margin_vs_zero": 0.70}
    show("control passes -> False", base_m(), c, False)

    print(f"battery complete: 7/7 branches on their pre-registered sides "
          f"(MI_MARGIN_MIN = {MI_MARGIN_MIN})")


if __name__ == "__main__":
    main()
