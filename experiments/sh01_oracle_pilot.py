"""SH.01 oracle pilot at the full envelope — the pre-registered launch gate.

sh_01_shelters_under_cold.py's PILOT RECORD (2026-08-19) ends with a binding
rule: "The registered run is deliberately NOT launched until an oracle pilot
at a larger budget (N scaled toward the full cpu<2h envelope, ~N=10000/arm)
shows the reference learning; if it cannot, the finding belongs to the
learning-core bakeoff (LC.04), not to this spec's ledger row." This script IS
that pilot, and nothing else: seed 90 (the pilot seed, disjoint from the
registered 0/1/2), the ORACLE arm (privileged working-hut direction in the
placebo slot) against the TWIN at the same budget, scored by the test's own
`_score` so the bar is byte-identical to the registered run's `ref_ok` gate.

DECISION RULE, restated from the test docstring before this run:
  ref z_shelter >= Z_MIN (3.0)  -> the registered run may launch (cpu<2h,
                                   3 seeds, via launch_detached/dispatch).
  ref z_shelter <  3.0          -> SH.01 stays unlaunched; the finding is a
                                   fourth instrument on the learning-core
                                   question (D10 evidence: the certified core
                                   cannot learn W0 behaviours at reachable
                                   envelopes even under privileged
                                   perception), NOT a ledger row and NOT a
                                   reason to grow the envelope again.

The artifact is written incrementally (started -> twin done -> final) so a
watcher can tell a crash from a long arm. No ledger write happens here — a
pilot is rig calibration, not evidence (the seed-90 precedent: PG.6, SM.01,
PS.02, this spec's own v1/v2 pilots).
"""
import json
import os
import sys
import time

REPO = "/home/opc/jackthelearner"
os.chdir(REPO)
sys.path.insert(0, REPO)

ART = "/data/sh01_oracle_pilot.json"
N = 10000
SEED = 90
T0 = time.time()


def _dump(stage: str, **kw) -> None:
    payload = {"stage": stage, "n_decisions": N, "seed": SEED,
               "elapsed_s": round(time.time() - T0, 1), **kw}
    tmp = ART + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, ART)
    print(f"[{payload['elapsed_s']:8.1f}s] {stage} {kw if kw else ''}",
          flush=True)


_dump("started")

from experiments.tests.sh_01_shelters_under_cold import (  # noqa: E402
    Z_MIN, _run_arm, _score)

twin = _run_arm(SEED, "twin", N)
_dump("twin_done", twin_wall_s=round(twin["wall_s"], 1),
      twin_steps=twin["optimiser_steps"], twin_lives=len(twin["lives"]))

orc = _run_arm(SEED, "oracle", N)
s = _score(orc, twin, N)

verdict = "ORACLE_LEARNS" if s["z_shelter"] >= Z_MIN else "ORACLE_CANNOT"
_dump("final", verdict=verdict, z_min=Z_MIN,
      score={k: round(float(v), 4) for k, v in s.items()},
      oracle_wall_s=round(orc["wall_s"], 1),
      oracle_steps=orc["optimiser_steps"], oracle_lives=len(orc["lives"]),
      oracle_hut_dec_all=int(sum(L["shelt"] for L in orc["lives"])),
      oracle_frozen=int(sum(L["frozen"] for L in orc["lives"])),
      twin_frozen=int(sum(L["frozen"] for L in twin["lives"])),
      physics_finite=min(orc["finite"], twin["finite"]))
