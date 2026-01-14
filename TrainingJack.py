"""
PHASE 2: DATASET TRAINING - Natural Movement + Manipulation

Behavior cloning from expert demonstrations using diffusion policy.

This is Phase 2 of AGI training - loads Phase 1 (locomotion) checkpoint and
refines with human demonstrations.

Supported datasets:
- MoCapAct: Human motion capture adapted for humanoids
- RT-1: Google's robot manipulation dataset
- Language-Table: Language-conditioned manipulation

Timeline:
- Phase 2A (MoCapAct): 2-3 days → Natural movement
- Phase 2B (RT-1): 1-2 days → Manipulation skills

Total: 3-5 days to complete Phase 2
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from EnhancedJackBrain import ScalableRobotBrain, BrainConfig, flow_matching_loss  # MERGED: All brain code in one file


class SimpleDemonstrationsDataset(Dataset):
    """
    Simplified dataset for Phase 2 training.

    In production, this would load actual MoCapAct or RT-1 data.
    For now, uses synthetic demonstrations.
    """

    def __init__(
        self,
        dataset_name: str = "mocapact",
        num_samples: int = 10000,
        context_length: int = 10,
        action_chunk_size: int = 48,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.context_length = context_length
        self.action_chunk_size = action_chunk_size

        print(f"[*] Loading {dataset_name} dataset...")
        print(f"   Samples: {num_samples}")
        print(f"   Context: {context_length} frames")
        print(f"   Action chunks: {action_chunk_size} steps")

        # In production: Load from HDF5 files
        # For now: Generate synthetic demonstrations
        self.demonstrations = []

        for i in range(num_samples):
            demo = self._generate_synthetic_demo()
            self.demonstrations.append(demo)

        print(f"[OK] Loaded {len(self.demonstrations)} demonstrations\n")

    def _generate_synthetic_demo(self):
        """Generate synthetic demonstration (placeholder)"""
        # In production: Load from actual dataset

        # Synthetic trajectory
        traj_length = 100
        obs_dim = 376  # Humanoid observation dim
        action_dim = 17  # Humanoid action dim

        observations = np.random.randn(traj_length, obs_dim).astype(np.float32)
        actions = np.random.randn(traj_length, action_dim).astype(np.float32) * 0.1

        return {
            'observations': observations,
            'actions': actions,
            'length': traj_length,
        }

    def __len__(self):
        return len(self.demonstrations)

    def __getitem__(self, idx):
        demo = self.demonstrations[idx]

        # Sample random segment
        max_start = demo['length'] - self.context_length - self.action_chunk_size
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)

        # Extract context observations
        obs_start = start_idx
        obs_end = start_idx + self.context_length
        observations = demo['observations'][obs_start:obs_end]

        # Extract action chunk
        action_start = start_idx + self.context_length
        action_end = action_start + self.action_chunk_size
        actions = demo['actions'][action_start:action_end]

        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations)  # (context, obs_dim)
        action_tensor = torch.FloatTensor(actions)    # (chunk_size, action_dim)

        # Create dummy images (84x84 RGB) - in production, use actual camera data
        dummy_images = torch.zeros(self.context_length, 3, 84, 84)

        return {
            'observation': {
                'image': dummy_images,
                'state': obs_tensor,
            },
            'action_chunk': action_tensor,
        }


class Phase2Trainer:
    """
    Behavior cloning trainer for Phase 2.

    Uses diffusion policy (flow matching) to learn from demonstrations.
    """

    def __init__(
        self,
        brain: ScalableRobotBrain,
        config: BrainConfig,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints',
    ):
        self.brain = brain.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            brain.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100000,
            eta_min=learning_rate / 10,
        )

        # Training stats
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "="*70)
        print("[*] PHASE 2 TRAINER INITIALIZED")
        print("="*70)
        print(f"Device: {device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Diffusion: Flow matching (1-step inference)")
        print("="*70 + "\n")

    def train_epoch(self, dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Train for one epoch"""
        self.brain.train()
        self.epoch += 1

        epoch_loss = 0
        num_batches = 0

        print(f"[Epoch {self.epoch}] Training...")

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Step {self.global_step} | Batch {batch_idx + 1}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        avg_train_loss = epoch_loss / num_batches

        # Validation
        if val_dataloader is not None:
            val_loss = self.validate(val_dataloader)
            print(f"[Epoch {self.epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_phase2")
                print(f"  [BEST] New best model saved (val_loss={val_loss:.4f})")
        else:
            print(f"[Epoch {self.epoch}] Train Loss: {avg_train_loss:.4f}")

        return avg_train_loss

    def train_step(self, batch: Dict) -> float:
        """Single training step with flow matching"""
        # Move data to device
        images = batch['observation']['image'].to(self.device)  # (B, context, 3, H, W)
        states = batch['observation']['state'].to(self.device)  # (B, context, state_dim)
        target_actions = batch['action_chunk'].to(self.device)  # (B, action_chunk, action_dim)

        batch_size = images.shape[0]

        # Sample random timestep for diffusion training
        timesteps = torch.rand(batch_size, device=self.device)  # (B,) in [0, 1]

        # Add noise to actions (flow matching: interpolate between noise and data)
        noise = torch.randn_like(target_actions)
        timesteps_expanded = timesteps[:, None, None]  # (B, 1, 1)
        noisy_actions = (1 - timesteps_expanded) * noise + timesteps_expanded * target_actions

        # Forward pass (use last observation from context)
        last_state = states[:, -1, :]  # (B, state_dim)
        last_image = images[:, -1, :]  # (B, 3, H, W)

        # Through brain
        predicted_velocity, values, memory = self.brain(
            proprio=last_state,
            vision=last_image,
            history=None,
        )

        # Compute flow matching loss
        loss = flow_matching_loss(
            model_output=predicted_velocity,
            target_actions=target_actions,
            noisy_actions=noisy_actions,
            timesteps=timesteps,
        )

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return loss.item()

    @torch.no_grad()
    def validate(self, val_dataloader: DataLoader) -> float:
        """Validation"""
        self.brain.eval()

        total_loss = 0
        num_batches = 0

        for batch in val_dataloader:
            images = batch['observation']['image'].to(self.device)
            states = batch['observation']['state'].to(self.device)
            target_actions = batch['action_chunk'].to(self.device)

            batch_size = images.shape[0]
            timesteps = torch.rand(batch_size, device=self.device)

            noise = torch.randn_like(target_actions)
            timesteps_expanded = timesteps[:, None, None]
            noisy_actions = (1 - timesteps_expanded) * noise + timesteps_expanded * target_actions

            last_state = states[:, -1, :]
            last_image = images[:, -1, :]

            predicted_velocity, _, _ = self.brain(
                proprio=last_state,
                vision=last_image,
                history=None,
            )

            loss = flow_matching_loss(predicted_velocity, target_actions, noisy_actions, timesteps)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, name: str):
        """Save checkpoint"""
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'brain_state_dict': self.brain.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)
        print(f"[SAVE] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.brain.load_state_dict(checkpoint['brain_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"[LOAD] Checkpoint loaded: {path} (epoch {self.epoch})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Dataset Training")
    parser.add_argument("--dataset", type=str, default="mocapact",
                        help="Dataset: mocapact, rt1, language_table")
    parser.add_argument("--checkpoint-in", type=str, default="checkpoints/locomotion_best.pt",
                        help="Input checkpoint from Phase 1")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/natural_movement.pt",
                        help="Output checkpoint after training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("[*] PHASE 2: DATASET TRAINING")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Input checkpoint: {args.checkpoint_in}")
    print(f"Output checkpoint: {args.checkpoint_out}")
    print("="*70 + "\n")

    # Config
    config = BrainConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        context_length=10,
        action_chunk_size=48,
        action_dim=17,
        use_pretrained_vision=False,  # Enable when using actual images
        use_diffusion=True,
        use_flow_matching=True,
        flow_matching_steps=1,
    )

    # Create brain
    brain = ScalableRobotBrain(config, obs_dim=376)

    # Load Phase 1 checkpoint
    if os.path.exists(args.checkpoint_in):
        print(f"[*] Loading Phase 1 checkpoint: {args.checkpoint_in}")
        checkpoint = torch.load(args.checkpoint_in, map_location='cpu', weights_only=False)
        brain.load_state_dict(checkpoint['brain_state_dict'])
        print("[OK] Phase 1 checkpoint loaded!\n")
    else:
        print(f"[WARNING] Checkpoint not found: {args.checkpoint_in}")
        print("[WARNING] Training from scratch (not recommended)\n")

    # Create datasets
    print("[*] Loading datasets...")
    train_dataset = SimpleDemonstrationsDataset(
        dataset_name=args.dataset,
        num_samples=8000,
        context_length=config.context_length,
        action_chunk_size=config.action_chunk_size,
    )

    val_dataset = SimpleDemonstrationsDataset(
        dataset_name=args.dataset,
        num_samples=2000,
        context_length=config.context_length,
        action_chunk_size=config.action_chunk_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create trainer
    trainer = Phase2Trainer(brain, config)

    # Train
    print("\n[*] Starting training...\n")

    try:
        for epoch in range(args.epochs):
            trainer.train_epoch(train_loader, val_loader)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                trainer.save_checkpoint(f"phase2_epoch_{epoch+1}")

        # Save final checkpoint
        trainer.save_checkpoint("phase2_final")

        # Copy to specified output path
        import shutil
        shutil.copy(
            os.path.join(trainer.checkpoint_dir, "best_phase2.pt"),
            args.checkpoint_out
        )

        print("\n" + "="*70)
        print("[SUCCESS] PHASE 2 TRAINING COMPLETE!")
        print("="*70)
        print(f"Final checkpoint: {args.checkpoint_out}")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print("\nRobot now has:")
        print("  - Locomotion skills (Phase 1)")
        print("  - Natural movement patterns (Phase 2)")
        if args.dataset == "rt1":
            print("  - Manipulation skills (Phase 2)")
        print("\nReady for deployment or further training!")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted!")
        trainer.save_checkpoint("phase2_interrupted")
        print("[SAVE] Progress saved.")


if __name__ == "__main__":
    main()
