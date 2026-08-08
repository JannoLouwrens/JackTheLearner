"""T6.03 — cross-session persistence: save, restart, and Jack is still Jack.

GOAL.md: "What he learned yesterday — about the world and about his owner —
persists on disk, inspectable, across restarts." ME.2/ME.5/ME.8 proved the
individual memory stores survive; this is the INTEGRATION claim: the whole
companion — weights, mood, personality, long-term memories, monologue,
training progress — goes through CompanionPersistence.save_all() and comes
back in a genuinely fresh process via load_all().

The teeth:

  1. A REAL restart. Session 1 runs in a child process: builds the real
     UnifiedBrain (58M params, LLM off), lives a little (20 owner facts into
     CompanionMemory, 12 emotional updates, a renamed personality, monologue
     entries, a global_step), saves, and EXITS. Session 2 is a separate
     process seeded DIFFERENTLY (its virgin weights provably differ), which
     first answers as the null, then load_all()s and answers again. Anything
     it recalls can only have come from the file.
  2. Recall: top-1 CompanionMemory retrieval of each stored owner fact must
     be exact after restart (>= 0.90). The registered metric.
  3. Fidelity, component by component, each a separate 0/1 gate: weights
     bit-identical (sha256 over the state_dict) yet different pre-load; PAD
     vector restored to <= 1e-5 while the virgin PAD sat >= 1e-3 away; mood
     history refilled INTO the MoodHistory object (len + last entry, and
     .record() still works — a restore that replaces it with a bare list
     fails here); personality name/backstory; monologue entries; global_step.

NULL (fresh instance with no memory, per spec): session 2 BEFORE load_all —
same process, same queries, empty memory. Recall must be <= 0.05.

CONTROL (must fail): a truncated copy of the save file. load_all() must
RAISE, not shrug and produce a plausible half-Jack — "corrupted" is the
spec's falsifier, so silent acceptance of corruption fails the spec. A
mid-file byte-flip is also attempted and recorded as information only
(torch's zip CRC behaviour is not ours to pre-register).

Found and fixed on the way in (Persistence.py): list(MoodHistory) raised
TypeError inside _collect_emotional_state's try/except, which silently
dropped the ENTIRE emotional state (PAD included) from every save ever
written; and _apply_emotional_state replaced the MoodHistory object with a
plain list, killing .record() after any restore. Gates 3 exist so neither
can regress silently.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
PY = "/data/venvs/jackthelearner/bin/python"

N_FACTS = 20
TOPICS = ["kettle", "scarf", "ladder", "compass", "lantern", "notebook",
          "teapot", "cushion", "shovel", "mirror", "basket", "candle",
          "anchor", "wallet", "ribbon", "drum", "vase", "stool",
          "planter", "thermos", "doormat", "coaster", "kite", "apron"]
VALUES = ["teal", "crimson", "olive", "violet", "amber", "ivory", "slate",
          "coral", "bronze", "indigo"]
PLACES = ["shelf", "porch", "kitchen", "hallway", "attic", "garden"]

MIN_RECALL = 0.90         # recall_after_restart, the registered metric
NULL_MAX = 0.05           # fresh instance with no memory
MIN_GAP = 0.80
PAD_POST_TOL = 1e-5       # restored PAD must match the saved one
PAD_PRE_MIN = 1e-3        # virgin PAD must have been measurably elsewhere

_CACHE: dict = {}         # seed -> tmpdir/save_path, shared with _control

_COMMON = f"""
import sys, json, io, contextlib, hashlib, random
sys.path.insert(0, {str(REPO)!r})
import torch
torch.set_num_threads(2)

def build_brain(seed):
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
    torch.manual_seed(seed)
    cfg = UnifiedBrainConfig()
    cfg.llm_enabled = False
    cfg.enable_intrinsic_motivation = False
    with contextlib.redirect_stdout(io.StringIO()):
        return UnifiedBrain(cfg)

def digest(brain):
    sd = brain.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()

def pad_list(brain):
    return [float(x) for x in brain.emotional_state.pad_vector.detach().cpu()]

def snapshot(brain):
    emo = brain.emotional_state
    hist = list(emo.history.entries)
    return {{
        "digest": digest(brain),
        "pad": pad_list(brain),
        "hist_len": len(hist),
        "hist_last": hist[-1] if hist else None,
        "name": brain.personality.config.name,
        "backstory": brain.personality.config.backstory,
        "global_step": getattr(brain, "global_step", None),
        "n_monologue": len(brain.inner_monologue.thought_history),
        "n_memories": len(brain.memory.memories),
    }}

def answer(brain, queries):
    out = {{}}
    with contextlib.redirect_stdout(io.StringIO()):
        for topic, q in queries.items():
            got = brain.memory.recall(q, top_k=1)
            out[topic] = got[0] if got else None
    return out
"""

_SESSION1 = _COMMON + """
seed, tmpdir = int(sys.argv[1]), sys.argv[2]
vocab = json.load(open(tmpdir + "/config.json"))
rng = random.Random(1000 * seed + 3)
brain = build_brain(seed)

topics = rng.sample(vocab["topics"], vocab["n_facts"])
facts, queries = {}, {}
with contextlib.redirect_stdout(io.StringIO()):
    for t in topics:
        v, p = rng.choice(vocab["values"]), rng.choice(vocab["places"])
        facts[t] = "the owner keeps the %s %s near the %s" % (v, t, p)
        queries[t] = "what do you remember about the %s" % t
        brain.memory.add(facts[t], importance=0.5 + 0.5 * rng.random())

emo = brain.emotional_state
for i in range(12):
    emo.update(reward=(1.0 if i % 3 else -0.5), user_interaction=0.7, dt=5.0)
if len(emo.history.entries) == 0:
    emo.history.record(60.0, tuple(pad_list(brain)), "engaged", event="interaction")

brain.personality.config.name = "Jack-%d" % seed
brain.personality.config.backstory = "the Jack who survived restart %d" % seed
for i in range(5):
    brain.inner_monologue.thought_history.append((float(i), "thought %d" % i, "reflection"))
brain.global_step = 4200 + seed

from Persistence import CompanionPersistence, SaveConfig
p = CompanionPersistence(SaveConfig(save_dir=tmpdir, save_prefix="t603"))
path = p.save_all(brain, world_state={"jack_position": [1.0, 2.0, 0.0], "time_of_day": 14.5})

report = snapshot(brain)
report.update(save_path=path, facts=facts, queries=queries)
json.dump(report, open(tmpdir + "/report1.json", "w"))
"""

_SESSION2 = _COMMON + """
seed, tmpdir = int(sys.argv[1]), sys.argv[2]
r1 = json.load(open(tmpdir + "/report1.json"))
brain = build_brain(seed + 5000)          # different draw: virgin weights differ

pre = snapshot(brain)
null_answers = answer(brain, r1["queries"])   # the null: fresh instance, no memory
brain.global_step = 0

from Persistence import CompanionPersistence, SaveConfig
p = CompanionPersistence(SaveConfig(save_dir=tmpdir, save_prefix="t603"))
with contextlib.redirect_stdout(io.StringIO()):
    p.load_all(brain, r1["save_path"])

post = snapshot(brain)
# The restored history must still BE a MoodHistory: .record() must work.
brain.emotional_state.history.record(999.0, (0.0, 0.0, 0.0), "probe", event="post-restore")
post["record_still_works"] = len(brain.emotional_state.history.entries) == post["hist_len"] + 1

json.dump({"pre": pre, "post": post, "null_answers": null_answers,
           "answers": answer(brain, r1["queries"])},
          open(tmpdir + "/report2.json", "w"))
"""

_CORRUPT = _COMMON + """
seed, tmpdir, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
r1 = json.load(open(tmpdir + "/report1.json"))
src = r1["save_path"]
bad = tmpdir + "/corrupt_%s.pt" % mode
raw = open(src, "rb").read()
if mode == "truncate":
    open(bad, "wb").write(raw[: int(len(raw) * 0.6)])
else:
    mid = len(raw) // 2
    open(bad, "wb").write(raw[:mid] + bytes(64) + raw[mid + 64:])

brain = build_brain(seed + 9000)
from Persistence import CompanionPersistence, SaveConfig
p = CompanionPersistence(SaveConfig(save_dir=tmpdir, save_prefix="t603"))
raised = 0
err = ""
try:
    with contextlib.redirect_stdout(io.StringIO()):
        p.load_all(brain, bad)
except Exception as e:
    raised = 1
    err = type(e).__name__
json.dump({"raised": raised, "error": err},
          open(tmpdir + "/corrupt_%s.json" % mode, "w"))
"""


def _run_child(script: str, *args: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(script))
        path = f.name
    try:
        r = subprocess.run([PY, path, *args], cwd=REPO, capture_output=True,
                           text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"child failed rc={r.returncode}: {r.stderr[-800:]}")
    finally:
        Path(path).unlink(missing_ok=True)


def _pad_dev(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _experiment(seed: int) -> dict:
    tmpdir = tempfile.mkdtemp(prefix=f"t603_s{seed}_")
    _CACHE[seed] = tmpdir
    json.dump({"topics": TOPICS, "values": VALUES, "places": PLACES,
               "n_facts": N_FACTS}, open(f"{tmpdir}/config.json", "w"))
    _run_child(_SESSION1, str(seed), tmpdir)
    _run_child(_SESSION2, str(seed), tmpdir)

    r1 = json.load(open(f"{tmpdir}/report1.json"))
    r2 = json.load(open(f"{tmpdir}/report2.json"))
    pre, post = r2["pre"], r2["post"]

    facts = r1["facts"]
    recall = sum(r2["answers"][t] == facts[t] for t in facts) / len(facts)
    null_recall = sum(r2["null_answers"][t] == facts[t] for t in facts) / len(facts)

    hist_ok = (post["hist_len"] == r1["hist_len"] > 0
               and post["hist_last"] == r1["hist_last"]
               and post["record_still_works"])
    pers_ok = (post["name"] == r1["name"] == f"Jack-{seed}"
               and post["backstory"] == r1["backstory"]
               and pre["name"] != r1["name"])
    return {
        "recall_after_restart": round(recall, 4),
        "null_recall": round(null_recall, 4),
        "gap": round(recall - null_recall, 4),
        "weights_match": int(post["digest"] == r1["digest"]),
        "weights_differ_preload": int(pre["digest"] != r1["digest"]),
        "pad_dev_postload": round(_pad_dev(post["pad"], r1["pad"]), 8),
        "pad_dev_preload": round(_pad_dev(pre["pad"], r1["pad"]), 6),
        "hist_restored": int(hist_ok),
        "personality_restored": int(pers_ok),
        "gstep_restored": int(post["global_step"] == r1["global_step"] == 4200 + seed),
        "monologue_restored": int(post["n_monologue"] == r1["n_monologue"] == 5),
        "n_memories_restored": int(post["n_memories"] == r1["n_memories"] == N_FACTS),
    }


def _control(seed: int) -> dict:
    """A corrupted save must be rejected loudly, not half-loaded quietly."""
    tmpdir = _CACHE[seed]
    try:
        _run_child(_CORRUPT, str(seed), tmpdir, "truncate")
        _run_child(_CORRUPT, str(seed), tmpdir, "byteflip")
        trunc = json.load(open(f"{tmpdir}/corrupt_truncate.json"))
        flip = json.load(open(f"{tmpdir}/corrupt_byteflip.json"))
        return {"truncated_load_raised": trunc["raised"],
                "byteflip_load_raised_info": flip["raised"]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _CACHE.pop(seed, None)


def _check(m: dict, c: dict) -> bool:
    return (m["recall_after_restart"] >= MIN_RECALL
            and m["null_recall"] <= NULL_MAX
            and m["gap"] >= MIN_GAP
            and m["weights_match"] == 1.0
            and m["weights_differ_preload"] == 1.0
            and m["pad_dev_postload"] <= PAD_POST_TOL
            and m["pad_dev_preload"] >= PAD_PRE_MIN
            and m["hist_restored"] == 1.0
            and m["personality_restored"] == 1.0
            and m["gstep_restored"] == 1.0
            and m["monologue_restored"] == 1.0
            and m["n_memories_restored"] == 1.0
            and c["truncated_load_raised"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T6.03"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    print(run())
