"""
UPGRADED TRAINING SCRIPT FOR SCALABLE ROBOT BRAIN (2025 SOTA)
Now with: Diffusion Policy + Pretrained VLM + Open X-Embodiment Dataset

MAJOR UPGRADES:
✅ Diffusion Policy: Flow matching for 1-step inference
✅ Pretrained VLM: DINOv2 + SigLIP fusion (no training from scratch!)
✅ Multi-task Data: Open X-Embodiment (1M+ trajectories, 22 robot types)
✅ Continuous Actions: No more discretization artifacts
✅ 48-action chunks: Boston Dynamics style

Training progression:
Phase 1 (Week 1):  Train on Open X-Embodiment (diverse tasks)
Phase 2 (Week 2):  Fine-tune on simulation (Humanoid-v4)
Phase 3 (Week 3):  Add real robot data (100+ episo
des)
Phase 4 (Month 1): Deploy to real robot 🚀
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Tuple
from scalable_robot_brain import (
    ScalableRobotBrain,
    BrainConfig,
    flow_matching_loss,
)
from open_x_dataloader import create_openx_dataloader


# ==============================================================================
# WRAPPER: Convert Gym to Multimodal Observations
# ==============================================================================

class MultiModalWrapper(gym.ObservationWrapper):
    """
    Wraps standard gym env to provide multimodal observations.
    Progressive enablement: proprio → vision → touch → language
    """
    
    def __init__(
        self,
        env,
        enable_vision: bool = False,
        enable_touch: bool = False,
        enable_language: bool = False,
    ):
        super().__init__(env)
        self.enable_vision = enable_vision
        self.enable_touch = enable_touch
        self.enable_language = enable_language
        
        # Store original observation space
        self.proprio_space = env.observation_space
        
        print(f"\n🔧 Multimodal Configuration:")
        print(f"   Proprioception: ✓ (always enabled)")
        print(f"   Vision:         {'✓' if enable_vision else '✗'}")
        print(f"   Touch:          {'✓' if enable_touch else '✗'}")
        print(f"   Language:       {'✗' if enable_language else '✗'}\n")
    
    def observation(self, obs):
        """Convert gym obs to multimodal dict"""
        multimodal_obs = {
            'proprio': torch.FloatTensor(obs).unsqueeze(0),  # Add batch dim
        }
        
        if self.enable_vision:
            # TODO: Connect to actual camera
            # For now: placeholder (will render from mujoco later)
            multimodal_obs['vision'] = torch.zeros(1, 3, 84, 84)
        
        if self.enable_touch:
            # TODO: Connect to force sensors
            # For now: placeholder
            multimodal_obs['touch'] = torch.zeros(1, 10)
        
        if self.enable_language:
            # TODO: Parse text commands
            # For now: placeholder for "walk forward"
            multimodal_obs['language'] = None
        
        return multimodal_obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info


# ==============================================================================
# DIFFUSION POLICY TRAINER (Behavior Cloning + Flow Matching)
# ==============================================================================

class DiffusionPolicyTrainer:
    """
    Modern robot learning trainer using:
    - Behavior cloning (learn from demonstrations)
    - Diffusion policy (flow matching)
    - Multi-task training (Open X-Embodiment)

    This is the SOTA approach as of 2024-2025.
    No RL needed for initial training - just learn from data!
    """

    def __init__(
        self,
        brain: ScalableRobotBrain,
        config: BrainConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.brain = brain.to(device)
        self.config = config
        self.device = device

        # Optimizer (AdamW is standard for transformers)
        self.optimizer = torch.optim.AdamW(
            brain.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler (cosine annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100000,  # Total steps
            eta_min=learning_rate / 10,
        )

        # Training statistics
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        print(f"🎯 Diffusion Policy Trainer Initialized")
        print(f"   Device: {device}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: {weight_decay}\n")

    def train_epoch(self, dataloader, val_dataloader=None):
        """Train for one epoch on the dataset"""
        self.brain.train()
        self.epoch += 1

        epoch_loss = 0
        num_batches = 0

        print(f"📈 Epoch {self.epoch} - Training...")

        for batch_idx, batch in enumerate(dataloader):
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
            print(f"✓ Epoch {self.epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save("models/best_model.pt")
                print(f"  💾 Best model saved (val_loss={val_loss:.4f})")
        else:
            print(f"✓ Epoch {self.epoch} | Train Loss: {avg_train_loss:.4f}")

        return avg_train_loss

    def train_step(self, batch: Dict) -> float:
        """Single training step with flow matching"""
        # Move data to device
        images = batch['observation']['image'].to(self.device)  # (B, context, 3, H, W)
        states = batch['observation']['state'].to(self.device)  # (B, context, state_dim)
        target_actions = batch['action_chunk'].to(self.device)  # (B, action_chunk, action_dim)

        batch_size = images.shape[0]
        context_length = images.shape[1]

        # Sample random timestep for diffusion training
        timesteps = torch.rand(batch_size, device=self.device)  # (B,) in [0, 1]

        # Add noise to actions (flow matching: interpolate between noise and data)
        noise = torch.randn_like(target_actions)
        timesteps_expanded = timesteps[:, None, None]  # (B, 1, 1)
        noisy_actions = (1 - timesteps_expanded) * noise + timesteps_expanded * target_actions

        # Forward pass through brain (use last observation from context)
        # For simplicity, we'll use the last observation and state
        last_image = images[:, -1, :]  # (B, 3, H, W)
        last_state = states[:, -1, :]  # (B, state_dim)

        # Forward through brain
        predicted_velocity, values = self.brain.action_decoder(
            memory=self.brain.temporal_memory(
                self.brain.cross_modal_fusion(
                    torch.cat([
                        self.brain.proprio_proj(
                            self.brain.proprio_encoder(last_state)
                        ).unsqueeze(1),
                        self.brain.vision_proj(
                            self.brain.vision_encoder(last_image)
                        ).unsqueeze(1),
                    ], dim=1)
                )
            ),
            actions=noisy_actions,
            timesteps=timesteps,
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
    def validate(self, val_dataloader) -> float:
        """Validate on validation set"""
        self.brain.eval()

        total_loss = 0
        num_batches = 0

        for batch in val_dataloader:
            images = batch['observation']['image'].to(self.device)
            states = batch['observation']['state'].to(self.device)
            target_actions = batch['action_chunk'].to(self.device)

            batch_size = images.shape[0]
            timesteps = torch.rand(batch_size, device=self.device)

            # Add noise
            noise = torch.randn_like(target_actions)
            timesteps_expanded = timesteps[:, None, None]
            noisy_actions = (1 - timesteps_expanded) * noise + timesteps_expanded * target_actions

            # Forward pass
            last_image = images[:, -1, :]
            last_state = states[:, -1, :]

            predicted_velocity, _ = self.brain.action_decoder(
                memory=self.brain.temporal_memory(
                    self.brain.cross_modal_fusion(
                        torch.cat([
                            self.brain.proprio_proj(
                                self.brain.proprio_encoder(last_state)
                            ).unsqueeze(1),
                            self.brain.vision_proj(
                                self.brain.vision_encoder(last_image)
                            ).unsqueeze(1),
                        ], dim=1)
                    )
                ),
                actions=noisy_actions,
                timesteps=timesteps,
            )

            loss = flow_matching_loss(predicted_velocity, target_actions, noisy_actions, timesteps)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'brain_state_dict': self.brain.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)
        print(f"💾 Saved checkpoint to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.brain.load_state_dict(checkpoint['brain_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"📂 Loaded checkpoint from {path} (epoch {self.epoch})")


# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    print("="*70)
    print("🚀 UPGRADED ROBOT BRAIN TRAINING (2025 SOTA)")
    print("="*70)
    print("\n📋 Configuration:")

    # Configuration
    config = BrainConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        context_length=50,
        action_chunk_size=48,  # Boston Dynamics style
        action_dim=7,  # 7-DoF for manipulation (adjust for your robot)
        use_pretrained_vision=True,  # ENABLED!
        vlm_backbone="prismatic",
        use_diffusion=True,
        use_flow_matching=True,  # 1-step inference
        flow_matching_steps=1,
    )

    print(f"   Model: {config.d_model}D, {config.n_heads} heads, {config.n_layers} layers")
    print(f"   Action chunks: {config.action_chunk_size}")
    print(f"   Vision: Pretrained VLM ({config.vlm_backbone})")
    print(f"   Diffusion: Flow matching ({config.flow_matching_steps}-step)")

    # Create dataloaders
    print("\n📦 Loading Open X-Embodiment Dataset...")

    train_dataloader = create_openx_dataloader(
        data_path="./open_x_data",  # Change to actual path or HuggingFace name
        batch_size=32,
        num_workers=4,
        split="train",
        action_chunk_size=config.action_chunk_size,
        context_length=10,
        max_episodes=1000,  # Start with subset for testing
    )

    val_dataloader = create_openx_dataloader(
        data_path="./open_x_data",
        batch_size=32,
        num_workers=4,
        split="val",
        action_chunk_size=config.action_chunk_size,
        context_length=10,
        max_episodes=100,
    )

    # Create brain
    print("\n🧠 Initializing Robot Brain...")

    # Get observation dimension from first batch
    dummy_batch = next(iter(train_dataloader))
    obs_dim = dummy_batch['observation']['state'].shape[-1]

    brain = ScalableRobotBrain(config, obs_dim=obs_dim)

    # Count parameters
    total_params = sum(p.numel() for p in brain.parameters())
    trainable_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f}MB")

    # Create trainer
    print("\n🎯 Initializing Trainer...")
    trainer = DiffusionPolicyTrainer(
        brain=brain,
        config=config,
        learning_rate=1e-4,
        weight_decay=1e-4,
    )

    # Check for existing checkpoint
    checkpoint_path = "models/latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found checkpoint: {checkpoint_path}")
        response = input("Load it? (y/n): ")
        if response.lower() == 'y':
            trainer.load(checkpoint_path)

    # Train
    print("\n" + "="*70)
    print("🏋️  STARTING TRAINING")
    print("="*70)

    try:
        num_epochs = 100

        for epoch in range(trainer.epoch, num_epochs):
            # Train epoch
            train_loss = trainer.train_epoch(train_dataloader, val_dataloader)

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                trainer.save(f"models/checkpoint_epoch_{epoch+1}.pt")
                trainer.save(checkpoint_path)

        print("\n✅ Training completed!")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted!")
        trainer.save(checkpoint_path)
        print("💾 Progress saved.")

    print("\n" + "="*70)
    print("🎓 NEXT STEPS:")
    print("="*70)
    print("1. Fine-tune on your specific robot (sim or real)")
    print("2. Test in simulation environment")
    print("3. Deploy to real robot 🤖")
    print("4. Scale to language commands and complex tasks")
    print("="*70)


if __name__ == "__main__":
