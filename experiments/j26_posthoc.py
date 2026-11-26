"""J26 POST-HOC (ORDERS Next-up 1, 11-26): the four held-out R² again, kept
only to looks where at least one ray hits a floor food. Not registered; it
changes no line and no verdict of the frozen probe (`j26_probe.py`, untouched).
Filter applied to train AND test alike; same MLP, epochs, seeds via
j26_probe.probe. Raw looks from /data/jack-logs/j26/looks.

    python -m experiments.j26_posthoc FORM [FORM ...] > out.jsonl
"""
import json, sys, time
import numpy as np
from experiments import j26_probe as P


def stack_seen(recs):
    X, Y = {}, []
    for r in recs:
        f, y = P.forms(r)
        ok = np.isfinite(r["label"][:, 0])
        gid = r["gid"][ok].astype(np.int64)
        seen = np.isin(gid, r["foods"].astype(np.int64)).any(axis=1)
        for k, v in f.items():
            X.setdefault(k, []).append(v[seen])
        Y.append(y[seen])
    return {k: np.concatenate(v) for k, v in X.items()}, np.concatenate(Y)


def main():
    import torch
    torch.set_num_threads(1)
    looks = "/data/jack-logs/j26/looks"
    Xtr, Ytr = stack_seen(P.load(looks, P.TRAIN))
    Xte, Yte = stack_seen(P.load(looks, P.TEST))
    print(json.dumps({"n_train_seen": int(len(Ytr)), "n_test_seen": int(len(Yte))}), flush=True)
    for form in sys.argv[1:]:
        for s in P.SEEDS:
            t0 = time.process_time()
            r2 = P.probe(Xtr[form], Ytr, Xte[form], Yte, s)
            print(json.dumps({"form": form, "seed": s, "r2": round(r2, 4),
                              "cpu_s": round(time.process_time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
