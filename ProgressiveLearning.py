"""
PROGRESSIVE LEARNING SYSTEM FOR JACK THE WALKER
Train one skill at a time, watch it learn, save progress, add next skill

Training progression:
1. Learn to stand (stabilize)
2. Learn to walk forward
3. Learn to walk in any direction
4. Learn to run
5. Learn to navigate obstacles
6. Learn to manipulate objects
7. Learn language-conditioned tasks

Each stage builds on previous skills (continual learning)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Optional
import time
from datetime import datetime
import json

# Import your brain architecture
import sys
sys.path.append(os.path.dirname(__file__))
from JackBrain import ScalableRobotBrain, BrainConfig


# ==============================================================================
# TRAINING CURRICULUM - One skill at a time
# ==============================================================================

class TrainingCurriculum:
    """
    Define what skills to learn and in what order.
    Each stage has its own reward function and success criteria.
    """

    STAGES = {
        "1_stand": {
            "name": "Standing Balance",
            "description": "Learn to stand upright without falling",
            "episodes": 1000,
            "success_threshold": 500,  # Stay upright for 500 steps
            "reward_fn": "standing_reward",
        },
        "2_walk_forward": {
            "name": "Walk Forward",
            "description": "Learn to walk straight ahead",
            "episodes": 5000,
            "success_threshold": 1000,  # Walk 1000 steps forward
            "reward_fn": "walking_forward_reward",
        },
        "3_walk_direction": {
            "name": "Walk Any Direction",
            "description": "Learn to walk in commanded direction",
            "episodes": 10000,
            "success_threshold": 1500,
            "reward_fn": "directional_walking_reward",
        },
        "4_run": {
            "name": "Running",
            "description": "Learn to run (faster locomotion)",
            "episodes": 15000,
            "success_threshold": 2000,
            "reward_fn": "running_reward",
        },
        "5_obstacles": {
            "name": "Navigate Obstacles",
            "description": "Walk around obstacles and uneven terrain",
            "episodes": 20000,
            "success_threshold": 2500,
            "reward_fn": "obstacle_navigation_reward",
        },
    }

    @staticmethod
    def get_stage(stage_name: str) -> Dict:
        """Get stage configuration"""
        return TrainingCurriculum.STAGES.get(stage_name)

    @staticmethod
    def get_next_stage(current_stage: str) -> Optional[str]:
        """Get next stage in curriculum"""
        stage_keys = list(TrainingCurriculum.STAGES.keys())
        if current_stage not in stage_keys:
            return stage_keys[0]  # Start from beginning

        current_idx = stage_keys.index(current_stage)
        if current_idx < len(stage_keys) - 1:
            return stage_keys[current_idx + 1]
        return None  # Completed all stages


# ==============================================================================
# REWARD FUNCTIONS - One for each skill
# ==============================================================================

class RewardFunctions:
    """
    Different reward functions for different skills.
    Rewards guide what the robot learns.
    """

    @staticmethod
    def standing_reward(obs, action, next_obs, done, info):
        """Reward for standing upright"""
        # Extract torso height and orientation
        torso_height = next_obs[0]  # Z position
        torso_upright = next_obs[1]  # Orientation

        # Reward staying upright and at correct height
        height_reward = 1.0 if torso_height > 1.0 else 0.0
        upright_reward = 1.0 if abs(torso_upright) < 0.2 else 0.0

        # Penalty for falling
        fall_penalty = -10.0 if done and torso_height < 0.5 else 0.0

        # Small penalty for large actions (energy efficiency)
        action_penalty = -0.01 * np.sum(np.square(action))

        total_reward = height_reward + upright_reward + fall_penalty + action_penalty
        return total_reward

    @staticmethod
    def walking_forward_reward(obs, action, next_obs, done, info):
        """Reward for walking forward"""
        # Extract velocity (how fast moving forward)
        forward_velocity = next_obs[11]  # X velocity

        # Reward forward movement
        forward_reward = 5.0 * forward_velocity

        # Maintain upright
        torso_height = next_obs[0]
        upright_reward = 1.0 if torso_height > 1.0 else 0.0

        # Penalty for falling
        fall_penalty = -10.0 if done and torso_height < 0.5 else 0.0

        # Small penalty for large actions
        action_penalty = -0.01 * np.sum(np.square(action))

        total_reward = forward_reward + upright_reward + fall_penalty + action_penalty
        return total_reward

    @staticmethod
    def directional_walking_reward(obs, action, next_obs, done, info):
        """Reward for walking in commanded direction"""
        # Similar to forward walking but with directional component
        # (Would need to add goal direction to observation)
        return RewardFunctions.walking_forward_reward(obs, action, next_obs, done, info)

    @staticmethod
    def running_reward(obs, action, next_obs, done, info):
        """Reward for running (faster movement)"""
        forward_velocity = next_obs[11]

        # Higher reward for faster speeds
        running_reward = 10.0 * forward_velocity if forward_velocity > 2.0 else 0.0

        # Other components same as walking
        torso_height = next_obs[0]
        upright_reward = 1.0 if torso_height > 1.0 else 0.0
        fall_penalty = -10.0 if done and torso_height < 0.5 else 0.0
        action_penalty = -0.01 * np.sum(np.square(action))

        total_reward = running_reward + upright_reward + fall_penalty + action_penalty
        return total_reward

    @staticmethod
    def obstacle_navigation_reward(obs, action, next_obs, done, info):
        """Reward for navigating around obstacles"""
        # (Would require obstacle positions in observation)
        return RewardFunctions.walking_forward_reward(obs, action, next_obs, done, info)


# ==============================================================================
# SIMPLE PPO TRAINER - Watch it learn!
# ==============================================================================

class SimpleProgressiveTrainer:
    """
    Simple RL trainer that you can watch learn in real-time.
    Uses PPO algorithm (simple and stable).
    """

    def __init__(
        self,
        env_name: str = "Humanoid-v4",
        brain_config: Optional[BrainConfig] = None,
        render: bool = True,  # Show visualization
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.render = render

        # Create environment
        render_mode = "human" if render else None
        self.env = gym.make(env_name, render_mode=render_mode)

        # Get dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Create brain (your SOTA architecture!)
        if brain_config is None:
            brain_config = BrainConfig(
                d_model=256,  # Smaller for faster training
                n_heads=4,
                n_layers=3,
                context_length=10,
                action_chunk_size=1,  # Online control (no chunking yet)
                action_dim=self.action_dim,
                use_pretrained_vision=False,  # Not using vision yet
                use_diffusion=False,  # Start simple
            )

        self.brain = ScalableRobotBrain(brain_config, obs_dim=self.obs_dim).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=3e-4)

        # Training state
        self.current_stage = None
        self.episode_count = 0
        self.best_reward = -float('inf')

        # Checkpoints directory
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"\n🤖 PROGRESSIVE TRAINER INITIALIZED")
        print(f"   Environment: {env_name}")
        print(f"   Observation dim: {self.obs_dim}")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Device: {device}")
        print(f"   Render: {render}")
        print(f"   Ready to learn!\n")

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from brain"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Forward through brain (simplified for now)
            proprio_emb = self.brain.proprio_encoder(obs_tensor)
            proprio_proj = self.brain.proprio_proj(proprio_emb).unsqueeze(1)

            # Fusion and temporal
            fused = self.brain.cross_modal_fusion(proprio_proj)
            memory = self.brain.temporal_memory(fused)

            # Get action
            actions, values = self.brain.action_decoder(memory)

            # Extract continuous action (no more discretization!)
            action = actions[0, 0, :].cpu().numpy()  # First action chunk

        return action

    def compute_reward(self, obs, action, next_obs, done, info, reward_fn_name):
        """Compute reward based on current stage"""
        reward_fn = getattr(RewardFunctions, reward_fn_name)
        return reward_fn(obs, action, next_obs, done, info)

    def train_stage(self, stage_name: str):
        """Train one stage of the curriculum"""
        stage = TrainingCurriculum.get_stage(stage_name)
        if stage is None:
            print(f"❌ Stage {stage_name} not found!")
            return False

        self.current_stage = stage_name

        print("\n" + "="*70)
        print(f"🎯 STARTING STAGE: {stage['name']}")
        print("="*70)
        print(f"   Description: {stage['description']}")
        print(f"   Episodes: {stage['episodes']}")
        print(f"   Success: {stage['success_threshold']} steps")
        print(f"   Reward function: {stage['reward_fn']}")
        print("="*70 + "\n")

        # Training loop
        num_episodes = stage['episodes']
        success_threshold = stage['success_threshold']
        reward_fn_name = stage['reward_fn']

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            # Collect trajectory
            observations = []
            actions = []
            rewards = []

            while not done:
                # Get action from brain
                action = self.get_action(obs)

                # Step environment
                next_obs, env_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Compute custom reward for this stage
                reward = self.compute_reward(obs, action, next_obs, done, info, reward_fn_name)

                # Store
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)

                episode_reward += reward
                episode_length += 1

                obs = next_obs

                # Optional: Show in real-time
                if self.render:
                    time.sleep(0.01)  # Slow down for visualization

            # Train on this episode (simplified PPO)
            self.train_on_episode(observations, actions, rewards)

            # Track stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            self.episode_count += 1

            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])

                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Best: {self.best_reward:.2f}")

                # Save checkpoint if best
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.save_checkpoint(f"best_{stage_name}")

            # Check if stage completed
            if avg_length >= success_threshold and episode > 100:
                print(f"\n✅ STAGE COMPLETED! Achieved {avg_length:.0f} steps (target: {success_threshold})")
                self.save_checkpoint(f"completed_{stage_name}")
                return True

        print(f"\n⚠️  Stage not completed in {num_episodes} episodes")
        print(f"   Best length: {max(episode_lengths):.0f} / {success_threshold}")
        print(f"   Continue training? (increase episodes or tune rewards)")

        return False

    def train_on_episode(self, observations, actions, rewards):
        """Simple policy gradient update (REINFORCE)"""
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)

        # Compute returns (discounted rewards)
        returns = []
        G = 0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass
        proprio_emb = self.brain.proprio_encoder(obs_tensor)
        proprio_proj = self.brain.proprio_proj(proprio_emb).unsqueeze(1)
        fused = self.brain.cross_modal_fusion(proprio_proj)
        memory = self.brain.temporal_memory(fused)
        pred_actions, values = self.brain.action_decoder(memory)
        pred_actions = pred_actions[:, 0, :]  # First action chunk

        # Policy loss (MSE for continuous actions)
        policy_loss = ((pred_actions - actions_tensor) ** 2).mean()

        # Value loss
        value_loss = ((values.squeeze() - returns) ** 2).mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        self.optimizer.step()

    def save_checkpoint(self, name: str):
        """Save current training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")

        torch.save({
            'brain_state_dict': self.brain.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_stage': self.current_stage,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward,
            'timestamp': datetime.now().isoformat(),
        }, checkpoint_path)

        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, name: str):
        """Load previous training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")

        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.brain.load_state_dict(checkpoint['brain_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_stage = checkpoint['current_stage']
        self.episode_count = checkpoint['episode_count']
        self.best_reward = checkpoint['best_reward']

        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        print(f"   Stage: {self.current_stage}")
        print(f"   Episodes: {self.episode_count}")
        print(f"   Best reward: {self.best_reward:.2f}")

        return True

    def list_checkpoints(self):
        """List all saved checkpoints"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]

        if not checkpoints:
            print("No checkpoints found")
            return

        print("\n📦 Available Checkpoints:")
        for cp in sorted(checkpoints):
            path = os.path.join(self.checkpoint_dir, cp)
            checkpoint = torch.load(path, map_location='cpu')
            print(f"   {cp}")
            print(f"      Stage: {checkpoint.get('current_stage', 'unknown')}")
            print(f"      Episodes: {checkpoint.get('episode_count', 0)}")
            print(f"      Saved: {checkpoint.get('timestamp', 'unknown')}")

    def train_curriculum(self, start_stage: Optional[str] = None):
        """Train through entire curriculum"""
        if start_stage is None:
            start_stage = list(TrainingCurriculum.STAGES.keys())[0]

        current_stage = start_stage

        while current_stage is not None:
            success = self.train_stage(current_stage)

            if success:
                # Move to next stage
                next_stage = TrainingCurriculum.get_next_stage(current_stage)
                if next_stage is None:
                    print("\n🎉 ALL STAGES COMPLETED! Your robot is fully trained!")
                    break

                print(f"\n➡️  Moving to next stage: {next_stage}")
                current_stage = next_stage
            else:
                print(f"\n⚠️  Stage {current_stage} needs more training")
                break


# ==============================================================================
# MAIN - Start learning!
# ==============================================================================

def main():
    print("="*70)
    print("🚀 JACK THE WALKER - PROGRESSIVE LEARNING")
    print("="*70)
    print("\nThis will train your robot step-by-step:")
    print("1. First learn to stand")
    print("2. Then learn to walk")
    print("3. Then learn advanced skills")
    print("\nYou can watch it learn in REAL-TIME!")
    print("="*70 + "\n")

    # Create trainer
    trainer = SimpleProgressiveTrainer(
        env_name="Humanoid-v4",
        render=True,  # Set False for faster training without visualization
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # List existing checkpoints
    trainer.list_checkpoints()

    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Start training from beginning (stage 1: Standing)")
    print("2. Continue from checkpoint")
    print("3. Train specific stage")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        print("\n🎬 Starting training from stage 1...")
        trainer.train_curriculum(start_stage="1_stand")

    elif choice == "2":
        print("\nAvailable checkpoints:")
        trainer.list_checkpoints()
        checkpoint_name = input("\nEnter checkpoint name (without .pt): ").strip()
        if trainer.load_checkpoint(checkpoint_name):
            print("\n▶️  Continuing training...")
            trainer.train_curriculum(start_stage=trainer.current_stage)

    elif choice == "3":
        print("\nAvailable stages:")
        for key, stage in TrainingCurriculum.STAGES.items():
            print(f"   {key}: {stage['name']}")
        stage_name = input("\nEnter stage name: ").strip()
        print(f"\n🎯 Training stage: {stage_name}")
        trainer.train_stage(stage_name)

    print("\n✅ Training session complete!")
    print(f"   Total episodes: {trainer.episode_count}")
    print(f"   Checkpoints saved in: {trainer.checkpoint_dir}/")


if __name__ == "__main__":
    main()
