"""Probe (55th audit B2): Arm C's unthresholded recall@k on the CERTIFIED
stem-disjoint fixture, k in {1, 10, 50} — the number ME.11.F's premise stands
on ("Arm C retrieves top-50; pilot recall@10 was 1.000, so the answer is
present"). Reuses ME.11.C's own _load_model/_DenseIndex/_Prov so no private
code path can flatter the arm. Diagnostic only: writes no ledger row.

Run: /data/venvs/jackthelearner/bin/python scripts/probe_me11c_recall_at_k.py
"""
import json
import sys

sys.path.insert(0, "/home/opc/jackthelearner")

import numpy as np

from experiments.fixtures import paraphrase_eval as F
from experiments.tests.me_11_c_static_embeddings import (
    MODEL, _DenseIndex, _load_model, _Prov)

KS = (1, 10, 50)
SEEDS = (0, 1, 2)


def main():
    model = _load_model(MODEL)
    out = {}
    for seed in SEEDS:
        fx = F.build(seed)
        events = fx["events"]
        texts = [e["text"] for e in events]
        prov = _Prov(events)
        headline = [c for c in fx["cues"] if not c["ambiguous"]]
        idx = _DenseIndex(model, texts)
        hits = {k: 0 for k in KS}
        for c in headline:
            mask = prov.mask(c.get("speaker"), c.get("channel"))
            q = idx.embed_query(c["text"])
            sims = idx.mat @ q
            sims[~mask] = -np.inf
            order = np.argsort(-sims)
            gold = set(c["gold"])
            for k in KS:
                if gold & set(int(i) for i in order[:k]):
                    hits[k] += 1
        out[f"seed{seed}"] = {f"recall@{k}": round(hits[k] / len(headline), 4)
                              for k in KS}
        out[f"seed{seed}"]["n_cues"] = len(headline)
    for k in KS:
        vals = [out[f"seed{s}"][f"recall@{k}"] for s in SEEDS]
        out[f"mean_recall@{k}"] = round(float(np.mean(vals)), 4)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
