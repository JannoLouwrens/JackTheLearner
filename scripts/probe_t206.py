"""Probe kernel for T2.06 sizing — times the PRODUCTION configuration on the
target GPU before the real dispatch (the T2.04/B1 lesson: a cost measured on
the smoke's configuration is not a cost for the production configuration; the
local smoke measured ~7 s/step on this box's ARM CPU, which puts the full run
past 30 min on any credible GPU multiplier, so the probe is mandatory).

Run from the repo root on this box, DETACHED, with a labelled receipt
(the T2.05-probe scar: an unlabelled attempt receipt cost an iteration of
/data/tmp archaeology):

    setsid nohup env JACK_SPEC_ID=T2.06-probe \
      /data/venvs/jackthelearner/bin/python scripts/probe_t206.py \
      > /data/tmp/probe_t206.log 2>&1 &

Times, at UnifiedBrainConfig(llm_enabled=False, enable_semantic_anchors=True)
— the exact production build:
  - brain build cost
  - per-train-step cost of the shipped grounding loss (5 warmup + 25 timed,
    BATCH as the test declares)
  - full eval cost (two passes over 400 pairs)
Prints the JSON the dispatch arithmetic needs; the numbers go into the
_submit comment in t2_06_language_action_alignment.py, committed before
dispatch.
"""

import json
from pathlib import Path

from experiments.gpu import build_job, submit

JOB = r'''
import json, os, time
import numpy as np
import torch

import experiments.tests.t2_06_language_action_alignment as t
from UnifiedBrain import (UnifiedBrain, UnifiedBrainConfig,
                          SemanticActionAnchors,
                          compute_language_grounding_loss,
                          grounding_fallback_tokens)

device = "cuda" if torch.cuda.is_available() else "cpu"
out = {"gpu": (torch.cuda.get_device_name(0) if device == "cuda" else "cpu")}

names = list(SemanticActionAnchors.ACTION_CATEGORIES.keys())
syns = SemanticActionAnchors.ACTION_CATEGORIES
rng = np.random.RandomState(0)
cats = rng.randint(0, 8, size=512)
phrases = [syns[names[c]][rng.randint(len(syns[names[c]]))] for c in cats]
acts = np.stack([t.gen_action(int(c), rng) for c in cats])
states = 0.1 * rng.randn(512, 256)

torch.manual_seed(0)
t0 = time.time()
cfg = UnifiedBrainConfig(llm_enabled=False, enable_semantic_anchors=True)
model = UnifiedBrain(cfg).to(device)
out["build_s"] = round(time.time() - t0, 2)

S = torch.tensor(states, dtype=torch.float32, device=device)
A = torch.tensor(acts, dtype=torch.float32, device=device)
opt = torch.optim.Adam(model.parameters(), lr=t.LR)
g = torch.Generator().manual_seed(1)
model.train()

def step():
    idx = torch.randint(0, len(S), (t.BATCH,), generator=g)
    loss, _ = compute_language_grounding_loss(
        model, S[idx], A[idx], [phrases[i] for i in idx.tolist()])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

for _ in range(5):                    # warmup (cudnn autotune, allocator)
    step()
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
for _ in range(25):
    step()
if device == "cuda":
    torch.cuda.synchronize()
out["train_s_per_step"] = round((time.time() - t0) / 25, 4)

model.eval()
t0 = time.time()
with torch.no_grad():
    for _pass in range(2):
        toks = grounding_fallback_tokens(phrases[:400]).to(device)
        lang = model.language_encoder(toks)
        _sel, probs = model.semantic_anchors(lang)
        ae = model.semantic_anchors.encode_actions(A[:400])
if device == "cuda":
    torch.cuda.synchronize()
out["eval_s_400rows_2pass"] = round(time.time() - t0, 2)

json.dump(out, open(os.path.join(os.environ["JACK_OUT"], "probe206.json"),
                    "w"), indent=1)
print("PROBE", json.dumps(out), flush=True)
'''


def main():
    job = build_job(JOB)
    res = submit(job, prefer="kaggle", est_hours=0.1, timeout_s=1500,
                 fetch=["probe206.json"])
    if not res.ok:
        raise SystemExit(f"probe failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["probe206.json"]).read_text())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
