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
  4. What persists is a LEARNED state and a restored FUNCTION, not a
     round-tripped tensor. See "The 09-13 strengthening" below.

The 09-13 strengthening (Review FULL, Part 2 — strictly additive, no bar moved)
------------------------------------------------------------------------------
Re-examined at 36 days, the oldest PASS on the board and never reconsidered.
The claim this spec is cited for is GOAL.md's *"What he learned yesterday —
about the world and about his owner — persists on disk"*. Every conjunct above
was true of a session-1 brain **that never took an optimiser step**: its
`state_dict` was the initialisation. So `weights_match` certified that
`torch.save`/`torch.load` round-trips a tensor — which is T0.03's claim, made
one tier lower and 300 lines cheaper — and the word "learned" in the sentence
this spec answers had no referent anywhere in the file. A gate can be perfectly
sound and still not be about what its spec is about.

Two conjuncts are therefore ADDED (nothing existing is touched, so the passing
set can only shrink):

  train_moved_weights   session 1 trains on a fixed batch and its digest must
                        leave initialisation. There is now something learned to
                        persist, and `weights_match` compares against the
                        TRAINED digest rather than the virgin one.
  probe_dev_postload    the restored brain must reproduce session 1's loss on a
                        held-out probe batch to <= PROBE_TOL (RNG matched by
                        seeding immediately before each measurement), while the
                        virgin brain's own reading must sit >= PROBE_PRE_MIN
                        away (probe_dev_preload) and be >= PROBE_SEP_MIN times
                        further off (probe_sep_ratio). Bytes are not behaviour:
                        these are the first conjuncts here that read the
                        restored brain as a FUNCTION rather than as a file.

STATUS OF THE ROW, said plainly: T6.03 could not be re-bought on 2026-09-13.
The runner demoted it to BLOCKED — `dependencies not satisfied: T2.10 (FAIL)`
— which is the FIRST honest evaluation of its dependency state since T2.10 fell
on 09-01 under its own paraphrase conjunct. The 08-08 PASS had been rendering
green for twelve days on a dead dependency, because the board reports a stored
status and nothing re-checks a dependency after the fact. The code above is
verified to execute and to discriminate (seed 0, out-of-band, no ledger write:
train_loss 0.39336 -> 0.238613, train_moved_weights 1, probe_dev_postload
1.13e-6, probe_dev_preload 7.82e-3, all fourteen original conjuncts still 1),
but the CERTIFICATE is owed and is bought only when T2.10 is repaired.

`train_loss_first`/`train_loss_last` are recorded so the "learned" claim is
inspectable rather than asserted; `train_loss_fell` is gated so a training
block that silently no-ops cannot satisfy `train_moved_weights` with noise.

NULL (fresh instance with no memory, per spec): session 2 BEFORE load_all —
same process, same queries, empty memory. Recall must be <= 0.05.

CONTROL (must fail): a truncated copy of the save file AND a mid-file
64-byte zero-overwrite. load_all() must RAISE on both, not shrug and
produce a plausible half-Jack — "corrupted" is the spec's falsifier, so
silent acceptance of corruption fails the spec. The byteflip arm was
information-only in the first run (torch's raw zip behaviour accepted it);
Persistence now wraps the payload with a sha256 (jack-save-v1) verified in
load_all, so the byteflip gate is pre-registered as of that hardening.

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

# --- the 09-13 strengthening: a LEARNED state, and a restored FUNCTION ---
TRAIN_STEPS = 24          # enough that the loss falls, so "learned" has a
TRAIN_BS = 16             # referent, and cheap enough to keep this spec CPU
TRAIN_POOL = 256          # sliced per step (T1.08's loop)
TRAIN_LR = 3e-4           # T1.07/T1.08/TrainingPipeline recipe
TRAIN_WARMUP = 4          # short: 24 steps under a 100-step warmup learn nothing
PROBE_SEED = 77           # seeded immediately before every probe, so any
                          # sampling inside the loss is MATCHED across processes
# PROBE_TOL is a CALIBRATED constant and was set AFTER seeing a reading, which
# the LG.12 lesson (2026-09-13) says must then be checked against its own
# REACHABLE RANGE and not only against the bar. That range, measured on seed 0:
#   noise floor  1.13e-6  restored vs trained at BIT-IDENTICAL weights — pure
#                         inter-process float nondeterminism, not a restore
#                         defect (weights_match reads 1 on the same run)
#   signal       7.82e-3  virgin brain vs trained, i.e. what a restore that
#                         silently did nothing would score
# 1e-4 sits ~88x above the noise and ~78x below the signal, so the conjunct is
# falsifiable from both sides rather than decorative. A first pass at 1e-6 sat
# BELOW the noise floor and was un-clearable by construction — the LG.03 defect
# found on 09-12, caught here before it could be registered.
PROBE_TOL = 1e-4          # restored brain must reproduce the trained reading
PROBE_PRE_MIN = 1e-3      # the virgin brain must be measurably elsewhere
PROBE_SEP_MIN = 50.0      # ...and the restored reading must be at least this
                          # many times closer than the virgin one, so the pair
                          # cannot both collapse toward zero and still clear

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

def make_batch(cfg, n, gen_seed):
    \"\"\"A rank-8 tanh map — the construction T1.07/T1.08 use, identifiable by
    construction. Drawn from its OWN generator so both sessions build the
    byte-identical batch regardless of the seed their weights came from.\"\"\"
    g = torch.Generator().manual_seed(gen_seed)
    obs = torch.randn(n, cfg.obs_dim, generator=g)
    A = torch.randn(cfg.obs_dim, 8, generator=g) / (cfg.obs_dim ** 0.5)
    B = torch.randn(8, cfg.action_chunk_size * cfg.action_dim, generator=g)
    tgt = (torch.tanh(obs @ A) @ B).view(
        n, cfg.action_chunk_size, cfg.action_dim) * 0.3
    return obs, tgt

def probe_loss(brain, cfg):
    \"\"\"A deterministic reading of the brain AS A FUNCTION. The loss samples
    internally (flow matching), so the RNG is re-seeded immediately before the
    call: identical weights on identical inputs must give an identical number,
    and that is the whole point of the conjunct that reads it. eval() + no_grad
    per the dropout lesson.\"\"\"
    obs, tgt = make_batch(cfg, 8, {PROBE_SEED} + 1)
    was_training = brain.training
    brain.eval()
    try:
        torch.manual_seed({PROBE_SEED})
        with torch.no_grad():
            return float(brain.action_training_loss(obs, tgt)["loss"])
    finally:
        brain.train(was_training)

def train_a_little(brain, cfg):
    \"\"\"Give the save something LEARNED to carry. A fixed rank-8 pool, sliced
    per step — T1.07/T1.08's loop verbatim. (Re-feeding one identical batch
    object raises "backward through the graph a second time": the brain retains
    graph-carrying internal state across calls, so each step gets fresh slices.)
    \"\"\"
    pool_o, pool_t = make_batch(cfg, {TRAIN_POOL}, {PROBE_SEED} + 2)
    brain.train()
    opt, step_fn = brain.make_action_optimizer(
        lr={TRAIN_LR}, warmup_steps={TRAIN_WARMUP}, max_grad_norm=2.0)
    bs, losses = {TRAIN_BS}, []
    with contextlib.redirect_stdout(io.StringIO()):
        for step in range({TRAIN_STEPS}):
            i = (step * bs) % ({TRAIN_POOL} - bs)
            loss = brain.action_training_loss(pool_o[i:i+bs], pool_t[i:i+bs])["loss"]
            opt.zero_grad(); loss.backward(); step_fn()
            losses.append(float(loss))
    return losses
"""

_SESSION1 = _COMMON + """
seed, tmpdir = int(sys.argv[1]), sys.argv[2]
vocab = json.load(open(tmpdir + "/config.json"))
rng = random.Random(1000 * seed + 3)
brain = build_brain(seed)

# --- he learns something FIRST, so that there is something learned to persist.
# Before the diary/mood/name writes, matching T1.07/T1.08's arm(): those train
# straight off construction, and this spec is not the place to discover a new
# interaction between training and a populated companion state. ---
from UnifiedBrain import UnifiedBrainConfig as _UBC
_cfg = _UBC()
digest_init = digest(brain)
train_losses = train_a_little(brain, _cfg)
probe_trained = probe_loss(brain, _cfg)

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
report.update(save_path=path, facts=facts, queries=queries,
              digest_init=digest_init, train_losses=train_losses,
              probe_trained=probe_trained)
json.dump(report, open(tmpdir + "/report1.json", "w"))
"""

_SESSION2 = _COMMON + """
seed, tmpdir = int(sys.argv[1]), sys.argv[2]
r1 = json.load(open(tmpdir + "/report1.json"))
brain = build_brain(seed + 5000)          # different draw: virgin weights differ

from UnifiedBrain import UnifiedBrainConfig as _UBC
_cfg = _UBC()
pre = snapshot(brain)
probe_pre = probe_loss(brain, _cfg)           # the virgin brain AS A FUNCTION
null_answers = answer(brain, r1["queries"])   # the null: fresh instance, no memory
brain.global_step = 0

from Persistence import CompanionPersistence, SaveConfig
p = CompanionPersistence(SaveConfig(save_dir=tmpdir, save_prefix="t603"))
with contextlib.redirect_stdout(io.StringIO()):
    p.load_all(brain, r1["save_path"])

post = snapshot(brain)
probe_post = probe_loss(brain, _cfg)          # the RESTORED brain, same reading
# The restored history must still BE a MoodHistory: .record() must work.
brain.emotional_state.history.record(999.0, (0.0, 0.0, 0.0), "probe", event="post-restore")
post["record_still_works"] = len(brain.emotional_state.history.entries) == post["hist_len"] + 1

json.dump({"pre": pre, "post": post, "null_answers": null_answers,
           "probe_pre": probe_pre, "probe_post": probe_post,
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

    tl = r1["train_losses"]
    return {
        # --- the 09-13 strengthening: a learned state, a restored function ---
        "train_loss_first": round(tl[0], 6),
        "train_loss_last": round(tl[-1], 6),
        "train_loss_fell": int(tl[-1] < tl[0]),
        # the saved weights left initialisation, so `weights_match` below is
        # now a claim about a TRAINED digest and not about a virgin one
        "train_moved_weights": int(r1["digest"] != r1["digest_init"]),
        "probe_trained": round(r1["probe_trained"], 8),
        "probe_dev_postload": round(abs(r2["probe_post"] - r1["probe_trained"]), 10),
        "probe_dev_preload": round(abs(r2["probe_pre"] - r1["probe_trained"]), 8),
        "probe_sep_ratio": round(abs(r2["probe_pre"] - r1["probe_trained"])
                                 / max(abs(r2["probe_post"] - r1["probe_trained"]),
                                       1e-12), 2),
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
                "byteflip_load_raised": flip["raised"]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _CACHE.pop(seed, None)


def _check(m: dict, c: dict) -> bool:
    return (
            # --- 09-13: what persisted was LEARNED, and it came back as a
            # FUNCTION and not only as bytes. Added conjuncts; nothing below
            # this block was touched and no bar moved. ---
            m["train_loss_fell"] == 1.0
            and m["train_moved_weights"] == 1.0
            and m["probe_dev_postload"] <= PROBE_TOL
            and m["probe_dev_preload"] >= PROBE_PRE_MIN
            and m["probe_sep_ratio"] >= PROBE_SEP_MIN
            # --- the original fourteen ---
            and m["recall_after_restart"] >= MIN_RECALL
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
            and c["truncated_load_raised"] == 1.0
            and c["byteflip_load_raised"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T6.03"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    print(run())
