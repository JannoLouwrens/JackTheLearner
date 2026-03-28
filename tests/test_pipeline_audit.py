"""Final audit of the training pipeline before Colab deployment."""
import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

results = []
def test(name, condition):
    status = "PASS" if condition else "FAIL"
    results.append((name, status))
    print(f"  [{status}] {name}")

print("=" * 60)
print("TRAINING PIPELINE - COMPLETE VERIFICATION")
print("=" * 60)

# Suppress init prints
old = sys.stdout; sys.stdout = io.StringIO()
from TrainingPipeline import TrainingPipeline, PipelineConfig
import tempfile
config = PipelineConfig(checkpoint_dir=tempfile.mkdtemp())
p = TrainingPipeline(config)
sys.stdout = old

# 1. INITIALIZATION
print("\n--- Initialization ---")
test("Model created", p.model is not None)
test("obs_proj created", p.obs_proj is not None)
test("log_std is Parameter", isinstance(p.log_std, torch.nn.Parameter))
test("log_std shape = action_dim", p.log_std.shape[0] == config.action_dim)
test("Replay buffer empty", len(p.replay) == 0)
test("EWC has empty fisher", len(p.ewc.fisher) == 0)
test("obs_count = 0", p.obs_count == 0)

# 2. OPTIMIZER INCLUDES EVERYTHING
print("\n--- Optimizer ---")
p.make_optimizer(0)
opt_ids = set()
for g in p.optimizer.param_groups:
    for param in g["params"]:
        opt_ids.add(id(param))
test("Model params in optimizer", all(id(x) in opt_ids for x in p.model.parameters()))
test("obs_proj params in optimizer", all(id(x) in opt_ids for x in p.obs_proj.parameters()))
test("log_std in optimizer", id(p.log_std) in opt_ids)
test("Adam eps=1e-5", p.optimizer.defaults["eps"] == 1e-5)

# 3. CHECKPOINT ROUND-TRIP
print("\n--- Checkpoint Round-Trip ---")
with torch.no_grad():
    p.obs_proj[0].weight.data.fill_(42.0)
    p.log_std.data.fill_(0.123)
p.obs_mean = torch.ones(376, device=p.device) * 5.0
p.obs_var = torch.ones(376, device=p.device) * 2.0
p.obs_count = 999
p.epoch = 7
p.global_step = 12345
p.save("audit_test")

ckpt = torch.load(os.path.join(config.checkpoint_dir, "audit_test.pt"), weights_only=False)
test("Checkpoint has model", "model" in ckpt)
test("Checkpoint has obs_proj", "obs_proj" in ckpt)
test("Checkpoint has log_std", "log_std" in ckpt)
test("Checkpoint has optimizer", "optimizer" in ckpt)
test("Checkpoint has obs_mean", "obs_mean" in ckpt)
test("Checkpoint has obs_var", "obs_var" in ckpt)
test("Checkpoint has obs_count", "obs_count" in ckpt)

# Reset and reload
with torch.no_grad():
    p.obs_proj[0].weight.data.fill_(0.0)
    p.log_std.data.fill_(0.0)
p.obs_count = 0; p.epoch = 0
p.load("audit_test")
test("obs_proj restored", p.obs_proj[0].weight.data.mean().item() > 40.0)
test("log_std restored", abs(p.log_std.data[0].item() - 0.123) < 0.001)
test("obs_mean restored", p.obs_mean.mean().item() > 4.0)
test("obs_count restored", p.obs_count == 999)
test("epoch restored", p.epoch == 7)

# 4. PPO UPDATE
print("\n--- PPO Update ---")
p.make_optimizer(2, lr=3.57e-5)
N = 64
rollout = {
    "states": torch.randn(N, 256, device=p.device),
    "actions": torch.randn(N, 17, device=p.device),
    "log_probs": torch.randn(N, device=p.device),
    "values": torch.randn(N, device=p.device),
    "rewards": torch.randn(N, device=p.device),
    "dones": torch.zeros(N, device=p.device),
}
w_before = p.model.action_head.locomotion_head.weight.data.clone()
metrics = p.rl_update(rollout)
w_after = p.model.action_head.locomotion_head.weight.data.clone()

test("pg_loss finite", np.isfinite(metrics["pg_loss"]))
test("vf_loss finite", np.isfinite(metrics["vf_loss"]))
test("entropy finite", np.isfinite(metrics["entropy"]))
test("Weights changed (gradients flowed)", (w_after - w_before).abs().sum().item() > 0)

# 5. OBS NORMALIZATION
print("\n--- Observation Normalization ---")
sys.stdout = io.StringIO()
p3 = TrainingPipeline(PipelineConfig(checkpoint_dir=tempfile.mkdtemp()))
sys.stdout = old
obs_big = np.random.randn(376).astype(np.float32) * 100
n = p3.normalize_obs(obs_big)
test("Normalized shape preserved", n.shape == (376,))
test("Normalized in [-10, 10]", np.all(np.abs(n) <= 10.001))
test("obs_count incremented", p3.obs_count == 1)

# 6. PROJECT OBS
print("\n--- project_obs ---")
test("256-dim passthrough", p.project_obs(torch.randn(1, 256, device=p.device)).shape == (1, 256))
test("376-dim projects to 256", p.project_obs(torch.randn(1, 376, device=p.device)).shape == (1, 256))
test("100-dim pads+projects", p.project_obs(torch.randn(1, 100, device=p.device)).shape == (1, 256))

# 7. REPLAY + EWC IN RL UPDATE
print("\n--- Anti-Forgetting ---")
p.replay.add({"state": torch.randn(256), "action": torch.randn(17),
              "next_state": torch.randn(256), "physics": torch.randn(10)}, phase=0)
p.replay.add({"state": torch.randn(256), "action": torch.randn(17),
              "next_state": torch.randn(256), "physics": torch.randn(10)}, phase=0)
def gen():
    for s in p.replay.sample(2):
        yield s["state"].unsqueeze(0).to(p.device), s["action"][:17].unsqueeze(0).to(p.device)
p.ewc.compute_fisher(gen, num_samples=2)
penalty = p.ewc.penalty()
test("EWC penalty >= 0", penalty.item() >= 0)
test("EWC penalty finite", np.isfinite(penalty.item()))
m2 = p.rl_update(rollout)
test("rl_update with replay+EWC works", np.isfinite(m2["pg_loss"]))

# 8. RL-ZOO3 HYPERPARAMS
print("\n--- RL-Zoo3 Humanoid Hyperparams ---")
test("gamma = 0.95", config.gamma == 0.95)
test("gae_lambda = 0.9", config.gae_lambda == 0.9)
test("clip_range = 0.3", config.clip_range == 0.3)
test("max_grad_norm = 2.0", config.max_grad_norm == 2.0)
test("entropy_coef = 0.002", config.entropy_coef == 0.002)
test("vf_coef = 0.43", config.vf_coef == 0.43)
test("n_epochs_ppo = 5", config.n_epochs_ppo == 5)
test("batch_size = 64", config.batch_size == 64)
test("n_steps = 512", config.n_steps == 512)

# SUMMARY
print()
print("=" * 60)
passed = sum(1 for _, s in results if s == "PASS")
failed = sum(1 for _, s in results if s == "FAIL")
print(f"RESULTS: {passed}/{len(results)} passed, {failed} failed")
print("=" * 60)
if failed:
    print()
    for n, s in results:
        if s == "FAIL":
            print(f"  FAILED: {n}")
sys.exit(0 if failed == 0 else 1)
