"""Thread scaling of the BC-ResNet-1 training step (N7-kws sizing, Next-up 1).

KNOWN_ANSWERS N7-kws says ~18 CPU-h a seed at 1 thread and "at 4 threads it may fit one
12 h Kaggle CPU session — unmeasured". This probe measures WALL ms a training step
(batch 100, the authors' model vendored verbatim, same step as `kws_cost_probe`) at
1 and 2 torch threads on the box, nice 19. 4 threads is NOT run here: the box's 4 cores
serve customers. Timing only: no learner trained, no rule, no ledger row. It sizes the
wall-clock of a Kaggle session only by extrapolation (different CPU, x86 vs ARM) —
a Kaggle probe is what settles the fit.

    python -m experiments.kws_thread_probe
"""
import json
import math
import time

import torch

from experiments.kws_cost_probe import load_model


def wall_step_ms(threads, n_steps=20, warm=4):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = load_model(1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3, momentum=0.9)
    lossf = torch.nn.CrossEntropyLoss()
    x = torch.randn(100, 1, 40, 101)
    y = torch.randint(0, 12, (100,))
    model.train()

    def step():
        opt.zero_grad()
        lossf(model(x), y).backward()
        opt.step()

    for _ in range(warm):
        step()
    w, c = time.perf_counter(), time.process_time()
    for _ in range(n_steps):
        step()
    return (time.perf_counter() - w) / n_steps * 1e3, (time.process_time() - c) / n_steps * 1e3


def main():
    out = {"torch": torch.__version__}
    for t in (1, 2):
        wall, cpu = wall_step_ms(t)
        out[f"t{t}"] = {"wall_ms": round(wall, 1), "cpu_ms": round(cpu, 1)}
    out["speedup_2v1_wall"] = round(out["t1"]["wall_ms"] / out["t2"]["wall_ms"], 2)
    spe = math.ceil(36923 / 100)
    out["wall_h_train_200ep_t2"] = round(200 * spe * out["t2"]["wall_ms"] / 3.6e6, 2)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
