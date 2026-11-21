"""J10 diagnosis (11-21): WHERE did the frozen walk row's seeds lose the walk?

Row J10-1 FAILED (s1 5,294 at 1M -> 585 at 2M, 10/10 fell; s0 5,834 / M 11.91
at 2M). The row evaluated only at 1M and 2M. Every segment kernel's output
keeps its `model.zip` (~20 MB; the 6 GB replay is not fetched), so the curve
at 0.5 / 1 / 1.5 / 2M can be read after the fact with the RULE'S OWN eval
(`humanoid_sac_row.evaluate`: 10 deterministic episodes, env seeds
10,000..10,009, centre-of-mass metres). Nothing here touches the frozen row,
its verdict or the ledger — it is a reading of checkpoints already paid for.

Validation BEFORE any new point is read: the box must reproduce the kernels'
own recorded evals at 1M and 2M (s0/s1/s2 seg1, s0/s1 seg3) to within 1% R;
if it does not, the new points are not reported (`validated: false`).

Also read per checkpoint: alpha = exp(log_ent_coef) (SAC auto), and the
critic's estimate min(Q1,Q2)(s0, pi(s0)) at each eval episode's first state
vs the discounted return (gamma 0.99) that episode actually earned — a cheap
look at whether the collapse comes with a critic blow-up.

    /data/venvs/jack-std/bin/python -m experiments.j10_ckpt_curve [CKPT_ROOT]

CKPT_ROOT defaults to /data/jack-logs/j10/ckpt_curve/<s?_seg?>/model.zip
(fetched with `kaggle kernels output <kernel> --file-pattern '^model\\.zip$'`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/data/jack-logs/j10/ckpt_curve")
REC = Path("/data/jack-logs/j10/sac_row")
GAMMA = 0.99
TOL = 0.01


def load_params_only(zip_path: Path):
    """The zips were written on Kaggle's Python 3.13; SBX pickles its whole
    policy, including cloudpickled jax functions (relu is a custom_jvp) whose
    3.13 bytecode SEGFAULTS when called under the box's 3.12 (measured 11-21).
    Unpickling is safe; calling is not. So: a fresh SAC built from the rule's
    own SAC_KW (same net, same spaces), and only the ARRAYS are copied in —
    actor params, critic params + targets, log_ent_coef. Faithfulness is not
    assumed: main() refuses to report unless the box reproduces the kernels'
    recorded evals."""
    import jax
    import numpy as np
    from sbx import SAC
    from experiments.humanoid_sac_row import SAC_KW, make_env
    src = SAC.load(str(zip_path))
    m = SAC("MlpPolicy", make_env(), seed=0, verbose=0, **SAC_KW)
    arr = lambda t: jax.tree_util.tree_map(lambda x: jax.numpy.asarray(np.asarray(x)), t)
    m.policy.actor_state = m.policy.actor_state.replace(params=arr(src.policy.actor_state.params))
    m.policy.qf_state = m.policy.qf_state.replace(
        params=arr(src.policy.qf_state.params), target_params=arr(src.policy.qf_state.target_params))
    m.ent_coef_state = m.ent_coef_state.replace(params=arr(src.ent_coef_state.params))
    m.num_timesteps = int(src.num_timesteps)
    return m


def read_one(zip_path: Path) -> dict:
    import numpy as np
    from sbx import SAC
    from experiments.humanoid_sac_row import evaluate, make_env
    model = load_params_only(zip_path)
    alpha = float(np.exp(np.asarray(model.ent_coef_state.params["log_ent_coef"])))
    starts = []
    pol = lambda o: model.predict(o, deterministic=True)[0]
    first = {"o": None}

    def logging_pol(o):
        if first["o"] is None:
            first["o"] = o
        return pol(o)

    # one pass per episode so the first obs of each is captured
    from experiments.humanoid_sac_row import EVAL_SEEDS
    eps = []
    for sd in EVAL_SEEDS:
        first["o"] = None
        ev = evaluate(logging_pol, seeds=(sd,))
        eps.append(ev["episodes"][0])
        starts.append(first["o"])
    R = sum(e["R"] for e in eps) / len(eps)
    M = sum(e["dx"] for e in eps) / max(sum(e["steps"] for e in eps), 1) * 1000.0
    q = None
    try:
        obs = np.stack(starts).astype(np.float32)
        act = np.stack([pol(o) for o in starts]).astype(np.float32)
        qs = model.policy.qf.apply(model.policy.qf_state.params, obs, act)
        q = float(np.asarray(qs).min(axis=0).mean())
    except Exception as e:  # reported, never hidden
        q = f"ERR {type(e).__name__}: {e}"[:200]
    # discounted return actually earned needs per-step rewards; evaluate() keeps
    # only totals, so report the cap a 1,000-step episode can earn at r/step = R/steps
    disc = []
    for e in eps:
        r_step = e["R"] / max(e["steps"], 1)
        disc.append(r_step * (1 - GAMMA ** e["steps"]) / (1 - GAMMA))
    return {"steps": int(model.num_timesteps), "R": R, "M": M,
            "fell": sum(e["fell"] for e in eps), "alpha": alpha, "q_start": q,
            "disc_return_flat": sum(disc) / len(disc),
            "ep_steps": [e["steps"] for e in eps]}


def main(root: Path = ROOT) -> dict:
    rows = {}
    for z in sorted(root.glob("s*_seg*/model.zip")):
        name = z.parent.name
        rows[name] = read_one(z)
        print(name, json.dumps(rows[name]), flush=True)
    checks = []
    for name, r in rows.items():
        rec = REC / f"{name}.json"
        ev = json.loads(rec.read_text()).get("eval") if rec.exists() else None
        if ev:
            rel = abs(r["R"] - ev["R"]) / max(abs(ev["R"]), 1.0)
            # POST-HOC (added 11-21 AFTER the 1% gate failed on 4/5): an episode the
            # kernel saw fall before step 500 must fall on the SAME step here. That
            # tests the load (wrong params change the first hundred steps); late
            # divergence is cross-CPU chaos in MuJoCo and is not tested by it.
            early = [(e["steps"], b) for e, b in zip(ev["episodes"], r["ep_steps"]) if e["steps"] < 500]
            checks.append({"ckpt": name, "kernel_R": ev["R"], "box_R": r["R"], "rel": rel,
                           "ok": rel <= TOL, "posthoc_early_falls": len(early),
                           "posthoc_early_same_step": sum(k == b for k, b in early)})
    validated = bool(checks) and all(c["ok"] for c in checks)
    out = {"validated": validated, "checks": checks, "rows": rows if validated else None,
           "rows_unvalidated": None if validated else rows}
    (root / "curve.json").write_text(json.dumps(out, indent=1) + "\n")
    print("VALIDATION", json.dumps(checks))
    print("VALIDATED" if validated else "NOT VALIDATED — new points not reported")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
