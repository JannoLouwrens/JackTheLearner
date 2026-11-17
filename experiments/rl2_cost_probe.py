"""RL² bandit row (PUBLISHED_PRIORS §6, n=100, k=5, published 78.7) — cost probe only.

Recipe, read verbatim from Duan et al. 2016 (arXiv:1611.02779v2, appendix A.1,
Table 1; source PDF from arxiv e-print, text via pypdf): TRPO, discount 0.99,
GAE λ 0.3, up to 1000 policy iterations (Fig. 2b's n=100 curves run to ~600),
GRU 256 units, mean KL 0.01, batch 250,000 steps per iteration (= 2,500 trials
of 100 pulls), separate policy and baseline GRUs, weight norm, ReLU.
Code: rllab + TensorFlow, never released for this experiment.

This probe times ONE forward+backward pass of a 256-unit GRU over a tenth
of a batch (250 sequences × 100 steps = 25,000 steps, kept small for the box's
RAM line; the full batch is 10× by linear scaling; input = one-hot action 5 +
reward + done + constant state 1 = 8), on one box core, nice 19. A TRPO
iteration costs at least: policy fwd+bwd, baseline fwd+bwd, ~10 conjugate-
gradient Fisher-vector products (each ≈ one fwd+bwd) and a line search
(several forwards) — so ≥ 12× the measured pass is a floor, not an estimate.
No learner is trained here; nothing is judged.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch

def main():
    torch.set_num_threads(1)
    B, T, H, D = 250, 100, 256, 8
    gru = torch.nn.GRU(D, H, batch_first=True)
    head = torch.nn.Linear(H, 5)
    x = torch.randn(B, T, D)
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ts = []
    for _ in range(reps):
        t0 = time.process_time()
        out, _ = gru(x)
        loss = head(out).logsumexp(-1).mean()
        loss.backward()
        ts.append(time.process_time() - t0)
    p = 10 * min(ts)   # scale 25k-step slice to the 250k-step batch
    floor_iter = 12 * p
    print(f"one fwd+bwd over 25k steps: {[round(t,2) for t in ts]} CPU-s -> 250k batch: {p:.1f} CPU-s")
    print(f"TRPO iteration floor (12 passes): {floor_iter:.0f} CPU-s")
    for iters in (600, 1000):
        print(f"  {iters} iterations x 1 seed: >= {floor_iter*iters/3600:.1f} CPU-h")

if __name__ == "__main__":
    main()
