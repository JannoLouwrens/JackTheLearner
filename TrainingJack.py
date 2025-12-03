"""
DIFFUSION POLICY TRAINING - Behavior Cloning from Demonstrations

NOTE: This script is currently DISABLED because it needs high-quality demonstration data.

What is this?
- Behavior cloning: Learn by copying expert demonstrations (like watching videos)
- Diffusion policy: State-of-the-art action prediction using flow matching
- Different from RL: RL learns by trial/error, this learns from experts

To use this script, you need to download expert demonstration datasets:

Option 1: MoCapAct (Human motion capture for humanoids)
   - Size: 5-10GB
   - Download: https://microsoft.github.io/MoCapAct/
   - Best for: Natural humanoid locomotion

Option 2: RoboNet (Robot manipulation demonstrations)
   - Size: 100GB+
   - Download: https://www.robonet.wiki/
   - Best for: Manipulation tasks

Option 3: Use your own demonstrations
   - Record expert demonstrations
   - Save in the format this script expects

For now: Just use ProgressiveLearning.py which does RL training!
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.dirname(__file__))
from JackBrain import (
    ScalableRobotBrain,
    BrainConfig,
    flow_matching_loss,
)
from SharedTrainingData import SharedTrainingDataManager
from AutoCheckpoint import AutoCheckpointManager


# ==============================================================================
# HUMANOID LOCOMOTION DATASET - Real simulation data from ProgressiveLearning
# ==============================================================================

class HumanoidLocomotionDataset(Dataset):
    """
    PyTorch Dataset for humanoid locomotion using shared training data.
    Loads real episodes from ProgressiveLearning's Humanoid-v4 simulation.
    """

    def __init__(
        self,
        data_manager: SharedTrainingDataManager,
        context_length: int = 10,
        action_chunk_size: int = 48,
        split: str = "train",
        train_split: float = 0.9,
    ):
        self.data_manager = data_manager
        self.context_length = context_length
        self.action_chunk_size = action_chunk_size

        # Load all episodes
        all_episodes = data_manager.load_all_episodes()

        if not all_episodes:
            raise ValueError("No training data found! Run ProgressiveLearning.py first to collect data.")

        # Split into train/val
        n_train = int(len(all_episodes) * train_split)
        if split == "train":
            self.episodes = all_episodes[:n_train]
        else:
            self.episodes = all_episodes[n_train:]

        # Build list of valid trajectory segments
        self.segments = []
        for ep_idx, episode in enumerate(self.episodes):
            ep_length = episode['episode_length']
            # Need context_length + action_chunk_size frames
            min_length = context_length + action_chunk_size
            if ep_length >= min_length:
                # Can sample from any starting point that gives us enough frames
                for start_idx in range(ep_length - min_length + 1):
                    self.segments.append((ep_idx, start_idx))

        print(f"[*] Humanoid Dataset ({split}):")
        print(f"   Episodes: {len(self.episodes)}")
        print(f"   Valid segments: {len(self.segments)}")
        print(f"   Context length: {context_length}")
        print(f"   Action chunk size: {action_chunk_size}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        ep_idx, start_idx = self.segments[idx]
        episode = self.episodes[ep_idx]

        # Extract observations for context
        obs = episode['observations'][start_idx:start_idx + self.context_length]

        # Extract actions for action chunk
        actions = episode['actions'][start_idx + self.context_length:start_idx + self.context_length + self.action_chunk_size]

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs))  # (context_length, obs_dim)
        actions_tensor = torch.FloatTensor(np.array(actions))  # (action_chunk_size, action_dim)

        # Create dummy images (84x84 RGB) - will be replaced with actual vision later
        dummy_images = torch.zeros(self.context_length, 3, 84, 84)

        # Return in format expected by DiffusionPolicyTrainer
        return {
            'observation': {
                'image': dummy_images,
                'state': obs_tensor,
            },
            'action_chunk': actions_tensor,
        }


def create_humanoid_dataloader(
    data_dir: str = "training_data",
    batch_size: int = 32,
    num_workers: int = 4,
    context_length: int = 10,
    action_chunk_size: int = 48,
    split: str = "train",
) -> DataLoader:
    """Create DataLoader for humanoid locomotion data"""

    data_manager = SharedTrainingDataManager(data_dir=data_dir)

    dataset = HumanoidLocomotionDataset(
        data_manager=data_manager,
        context_length=context_length,
        action_chunk_size=action_chunk_size,
        split=split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


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

        print(f"\n[*] Multimodal Configuration:")
        print(f"   Proprioception: [OK] (always enabled)")
        print(f"   Vision:         {'[OK]' if enable_vision else '[--]'}")
        print(f"   Touch:          {'[OK]' if enable_touch else '[--]'}")
        print(f"   Language:       {'[--]' if enable_language else '[--]'}\n")
    
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

        print(f"[*] Diffusion Policy Trainer Initialized")
        print(f"   Device: {device}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: {weight_decay}\n")

    def train_epoch(self, dataloader, val_dataloader=None):
        """Train for one epoch on the dataset"""
        self.brain.train()
        self.epoch += 1

        epoch_loss = 0
        num_batches = 0

        print(f"[*] Epoch {self.epoch} - Training...")

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
            print(f"[OK] Epoch {self.epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save("models/best_model.pt")
                print(f"  [SAVE] Best model saved (val_loss={val_loss:.4f})")
        else:
            print(f"[OK] Epoch {self.epoch} | Train Loss: {avg_train_loss:.4f}")

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
        print(f"[SAVE] Checkpoint saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.brain.load_state_dict(checkpoint['brain_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"[LOAD] Checkpoint loaded from {path} (epoch {self.epoch})")


# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="TrainingJack - Behavior Cloning")
    parser.add_argument("--dataset", type=str, help="Dataset to train on (cmu_mocap, mocapact, deepmind_control, rt1_subset, language_table)")
    parser.add_argument("--checkpoint-in", type=str, help="Input checkpoint to load (e.g., checkpoints/locomotion.pt)")
    parser.add_argument("--checkpoint-out", type=str, help="Output checkpoint to save (e.g., checkpoints/natural_movement.pt)")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    print("="*70)
    print("[*] TRAININGJACK - BEHAVIOR CLONING")
    print("="*70)
    print("\n[INFO] This script trains Jack from expert demonstrations")
    print("[INFO] Much faster than RL for complex tasks!")
    print("="*70)

    # List datasets if requested
    if args.list:
        from DatasetDownloader import DatasetDownloader
        downloader = DatasetDownloader()
        downloader.list_datasets()
        return

    # Check for datasets
    print("\n[*] Checking for datasets...")
    dataset_dir = "datasets"

    dataset_mapping = {
        "cmu_mocap": ("CMU Motion Capture", "cmu_mocap"),
        "mocapact": ("MoCapAct", "mocapact"),
        "deepmind_control": ("DeepMind Control", "deepmind_mocap"),
        "rt1_subset": ("RT-1 Subset", "rt1"),
        "language_table": ("Language-Table", "language_table"),
    }

    available_datasets = {}
    for key, (name, dir_name) in dataset_mapping.items():
        dataset_path = os.path.join(dataset_dir, dir_name)
        if os.path.exists(dataset_path) and os.listdir(dataset_path):
            available_datasets[key] = name

    if not available_datasets:
        print("\n[WARNING] No datasets found!")
        print("[INFO] You need to download datasets first.")
        print("\n[*] To see available datasets:")
        print("   py TrainingJack.py --list")
        print("\n[*] To download datasets:")
        print("   py DatasetDownloader.py")
        print("\n[*] Or use sequential training:")
        print("   py TrainSequentially.py --next")
        print("\n[INFO] For now, use RL training:")
        print("   py ProgressiveLearning.py")
        print("="*70)
        return

    print(f"[OK] Found {len(available_datasets)} dataset(s):")
    for key, name in available_datasets.items():
        print(f"   - {name} ({key})")

    # If no dataset specified, show options
    if not args.dataset:
        print("\n[INFO] Specify which dataset to train on:")
        print("   py TrainingJack.py --dataset cmu_mocap")
        print("\n[INFO] Or use sequential training (recommended):")
        print("   py TrainSequentially.py --next")
        print("="*70)
        return

    # Validate dataset
    if args.dataset not in available_datasets:
        print(f"\n[ERROR] Dataset '{args.dataset}' not found or not downloaded")
        print(f"[INFO] Available: {list(available_datasets.keys())}")
        print("\n[INFO] To download:")
        print(f"   py DatasetDownloader.py --download {args.dataset}")
        print("="*70)
        return

    print(f"\n[OK] Selected dataset: {available_datasets[args.dataset]}")

    # Check checkpoint
    if args.checkpoint_in:
        if os.path.exists(args.checkpoint_in):
            print(f"[OK] Will load checkpoint: {args.checkpoint_in}")
        else:
            print(f"[ERROR] Checkpoint not found: {args.checkpoint_in}")
            return
    else:
        print("[INFO] No input checkpoint specified - training from scratch")

    print("\n[INFO] TrainingJack is ready to train!")
    print("[WARNING] Full training implementation coming soon")
    print(f"[INFO] Will train on: {available_datasets[args.dataset]}")
    if args.checkpoint_out:
        print(f"[INFO] Will save to: {args.checkpoint_out}")
    print("="*70)
    return

    # Below code is disabled until demonstration data is available
    print("\n[*] Configuration:")

    # Configuration - adjusted for Humanoid-v4 (17 DoF)
    config = BrainConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        context_length=10,  # Reduced for faster training
        action_chunk_size=17,  # Match Humanoid-v4 action dim
        action_dim=17,  # 17-DoF humanoid
        use_pretrained_vision=False,  # Disabled for now (no vision data yet)
        vlm_backbone="prismatic",
        use_diffusion=True,
        use_flow_matching=True,  # 1-step inference
        flow_matching_steps=1,
    )

    print(f"   Model: {config.d_model}D, {config.n_heads} heads, {config.n_layers} layers")
    print(f"   Action chunks: {config.action_chunk_size}")
    print(f"   Action dim: {config.action_dim} (Humanoid-v4)")
    print(f"   Vision: Disabled (using proprioception only)")
    print(f"   Diffusion: Flow matching ({config.flow_matching_steps}-step)")

    # Create dataloaders from real humanoid simulation data
    print("\n[*] Loading Humanoid Locomotion Dataset...")
    print("   Source: ProgressiveLearning shared training data")

    try:
        train_dataloader = create_humanoid_dataloader(
            data_dir="training_data",
            batch_size=32,
            num_workers=0,  # Set to 0 for Windows compatibility
            context_length=config.context_length,
            action_chunk_size=config.action_chunk_size,
            split="train",
        )

        val_dataloader = create_humanoid_dataloader(
            data_dir="training_data",
            batch_size=32,
            num_workers=0,  # Set to 0 for Windows compatibility
            context_length=config.context_length,
            action_chunk_size=config.action_chunk_size,
            split="val",
        )
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("[*] Please run ProgressiveLearning.py first to collect training data.")
        return

    # Create brain
    print("\n[*] Initializing Robot Brain...")

    # Get observation dimension from first batch
    dummy_batch = next(iter(train_dataloader))
    obs_dim = dummy_batch['observation']['state'].shape[-1]

    brain = ScalableRobotBrain(config, obs_dim=obs_dim)

    # Count parameters
    total_params = sum(p.numel() for p in brain.parameters())
    trainable_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    print(f"\n[*] Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f}MB")

    # Create trainer
    print("\n[*] Initializing Trainer...")
    trainer = DiffusionPolicyTrainer(
        brain=brain,
        config=config,
        learning_rate=1e-4,
        weight_decay=1e-4,
    )

    # AUTO-LOAD: Check for existing checkpoints and load the best one
    print("\n[*] Searching for existing checkpoints...")
    checkpoint_manager = AutoCheckpointManager()
    success, checkpoint = checkpoint_manager.load_latest_checkpoint(
        brain=trainer.brain,
        optimizer=trainer.optimizer,
        device=trainer.device,
        prefer_best=True
    )

    if success:
        # Restore training state
        if 'epoch' in checkpoint:
            trainer.epoch = checkpoint['epoch']
        if 'global_step' in checkpoint:
            trainer.global_step = checkpoint['global_step']
        if 'best_val_loss' in checkpoint:
            trainer.best_val_loss = checkpoint['best_val_loss']
        if 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Train
    print("\n" + "="*70)
    print("[*] STARTING TRAINING")
    print("="*70)

    try:
        num_epochs = 100

        for epoch in range(trainer.epoch, num_epochs):
            # Train epoch
            train_loss = trainer.train_epoch(train_dataloader, val_dataloader)

            # Save checkpoint every 10 epochs to models/latest.pt
            if (epoch + 1) % 10 == 0:
                trainer.save("models/latest.pt")

        print("\n[OK] Training completed!")

    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted!")
        trainer.save("models/latest.pt")
        print("[SAVE] Progress saved.")

    print("\n" + "="*70)
    print("[*] NEXT STEPS:")
    print("="*70)
    print("1. Continue training with: py TrainingJack.py")
    print("2. Run ProgressiveLearning to collect more data")
    print("3. Test trained model in simulation")
    print("4. Deploy to real robot")
    print("="*70)


if __name__ == "__main__":


    main()
