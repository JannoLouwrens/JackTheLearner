"""A continuation round: rung L only, carrying candidates that climbed in a named
parent round (ruling 123; ORDERS J18, round 1-L).

`validate_spec` in `experiments/forge/__init__.py` stays untouched: that file is
in round 1's judge lock (`rounds/r1/judge_lock.json`, commit 2eb2c0f4), and moving
it would make round 1's results unreadable (`check_lock` refuses). This module
holds the continuation rule instead. A spec WITH `parent` goes through here; a
spec without one goes to `validate_spec` unchanged, so a round with no parent and
no S rung still refuses there.

What differs from `validate_spec`, and why:
- no S rung required, and only L allowed: the parent's S and M already ran;
- no control-source candidate required: the parent's control (plain variation at
  the same budget) ran at the parent's rungs, so it must be NAMED instead
  (`parent.control_source`) together with the parent's judge-lock commit;
- at most MAX_LONG candidates (FORGE.md: rung L, at most 2 + the baseline).
Every other clause (ids, sources, priors, twins, distinct seeds, the outcome
requiring eating with a frozen metric hash, a numeric budget) is the same.
"""
from experiments.forge import MAX_LONG, SOURCES, Refused, validate_spec


def validate_round(spec: dict) -> None:
    """Validate any round spec: continuation rounds here, all others by validate_spec."""
    if "parent" not in spec:
        validate_spec(spec)
        return
    validate_continuation(spec)


def validate_continuation(spec: dict) -> None:
    def need(cond, msg):
        if not cond:
            raise Refused(f"round spec: {msg}")

    for k in ("round", "question", "baseline", "twins", "candidates", "rungs", "outcome",
              "budget", "parent"):
        need(k in spec, f"missing '{k}'")
    parent = spec["parent"]
    need(isinstance(parent, dict), "parent must name the parent round's lock and control source")
    need(isinstance(parent.get("round"), str) and parent["round"].strip()
         and parent["round"] != spec["round"], "parent.round must name another round")
    need(isinstance(parent.get("lock"), str) and len(parent["lock"]) >= 7,
         "parent.lock (the parent round's judge-lock commit) is missing")
    need(isinstance(parent.get("control_source"), str) and parent["control_source"].strip(),
         "parent.control_source (where the parent's control ran) is missing")

    cands = spec["candidates"]
    ids = [c.get("id") for c in cands]
    need(len(ids) >= 1 and all(isinstance(i, str) and i for i in ids), "every candidate needs an id")
    reserved = [spec["baseline"], *spec["twins"]]
    need(len(set(ids + reserved)) == len(ids) + len(reserved),
         "candidate, baseline and twin ids must all differ")
    for c in cands:
        need(c.get("source") in SOURCES, f"{c['id']}: source must be one of {SOURCES}")
        need(isinstance(c.get("prior"), str) and c["prior"].strip(),
             f"{c['id']}: no prior (law 3a)")
        need(c.get("twin") in spec["twins"], f"{c['id']}: twin must be one of the round's twins")
    need(len(cands) <= MAX_LONG, f"rung L carries at most {MAX_LONG} candidates")

    need(set(spec["rungs"]) == {"L"}, "a continuation round runs rung L only")
    r = spec["rungs"]["L"]
    need(int(r.get("decisions", 0)) > 0, "rung L: decisions > 0")
    seeds = r.get("seeds", [])
    need(len(seeds) >= 1 and len(set(seeds)) == len(seeds), "rung L: distinct seeds")

    out = spec["outcome"]
    need(out.get("requires_eating") is True, "the outcome must require eating")
    need(isinstance(out.get("metric_sha256"), str) and len(out["metric_sha256"]) == 64,
         "outcome.metric_sha256 (the frozen metric code hash) is missing")
    for k in ("core_hours", "kernel_hours"):
        need(isinstance(spec["budget"].get(k), (int, float)) and spec["budget"][k] >= 0,
             f"budget.{k} must be a number")
