"""
SHARED TRAINING DATA MANAGER
Automatically shares training experiences between ProgressiveLearning and TrainingJack.

Both scripts write their training episodes to a shared directory.
Both scripts automatically load and train on ALL available data.

This enables:
- Continuous learning across different training sessions
- Data accumulation over time
- Automatic use of latest training data
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
import pickle


class SharedTrainingDataManager:
    """
    Manages shared training data between different training scripts.

    Features:
    - Automatically saves training episodes
    - Automatically loads all available data
    - Tracks data statistics and metadata
    - Supports incremental data addition
    """

    def __init__(self, data_dir: str = "training_data"):
        """
        Args:
            data_dir: Directory to store shared training data
        """
        self.data_dir = data_dir
        self.episodes_dir = os.path.join(data_dir, "episodes")
        self.metadata_path = os.path.join(data_dir, "metadata.json")

        # Create directories
        os.makedirs(self.episodes_dir, exist_ok=True)

        # Load or create metadata
        self.metadata = self._load_metadata()

        print(f"[*] Shared Training Data Manager initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Total episodes: {self.metadata['total_episodes']}")
        print(f"   Total transitions: {self.metadata['total_transitions']}")

    def _load_metadata(self) -> Dict:
        """Load metadata about training data"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'total_episodes': 0,
                'total_transitions': 0,
                'last_episode_id': 0,
                'training_sessions': [],
                'created_at': datetime.now().isoformat(),
            }

    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_episode(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
        infos: List[Dict],
        source: str = "unknown"
    ) -> str:
        """
        Save a single training episode to shared storage.

        Args:
            observations: List of observations
            actions: List of actions taken
            rewards: List of rewards received
            dones: List of done flags
            infos: List of info dicts
            source: Name of training script (e.g., "ProgressiveLearning", "TrainingJack")

        Returns:
            episode_id: Unique ID for this episode
        """
        # Generate episode ID
        episode_id = self.metadata['last_episode_id'] + 1
        self.metadata['last_episode_id'] = episode_id

        # Create episode data
        episode_data = {
            'episode_id': episode_id,
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'infos': infos,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'episode_length': len(observations),
            'total_reward': float(sum(rewards)),  # Convert to native Python float
        }

        # Save episode to file
        episode_path = os.path.join(self.episodes_dir, f"episode_{episode_id:06d}.pkl")
        with open(episode_path, 'wb') as f:
            pickle.dump(episode_data, f)

        # Update metadata
        self.metadata['total_episodes'] += 1
        self.metadata['total_transitions'] += len(observations)
        self.metadata['training_sessions'].append({
            'episode_id': int(episode_id),
            'source': source,
            'timestamp': episode_data['timestamp'],
            'length': int(episode_data['episode_length']),
            'reward': float(episode_data['total_reward']),  # Ensure it's native Python float
        })
        self._save_metadata()

        return f"episode_{episode_id:06d}"

    def load_all_episodes(self, max_episodes: Optional[int] = None) -> List[Dict]:
        """
        Load all available training episodes.

        Args:
            max_episodes: Maximum number of episodes to load (loads latest N)

        Returns:
            List of episode dictionaries
        """
        # Get all episode files
        episode_files = sorted([
            f for f in os.listdir(self.episodes_dir)
            if f.startswith('episode_') and f.endswith('.pkl')
        ])

        # Load latest N episodes if max_episodes specified
        if max_episodes is not None and len(episode_files) > max_episodes:
            episode_files = episode_files[-max_episodes:]

        # Load all episodes
        episodes = []
        for episode_file in episode_files:
            episode_path = os.path.join(self.episodes_dir, episode_file)
            with open(episode_path, 'rb') as f:
                episodes.append(pickle.load(f))

        print(f"[*] Loaded {len(episodes)} episodes from shared data")
        if episodes:
            total_transitions = sum(ep['episode_length'] for ep in episodes)
            avg_reward = np.mean([ep['total_reward'] for ep in episodes])
            print(f"   Total transitions: {total_transitions}")
            print(f"   Average reward: {avg_reward:.2f}")

        return episodes

    def load_recent_episodes(self, n_episodes: int = 100) -> List[Dict]:
        """
        Load the N most recent episodes.

        Args:
            n_episodes: Number of recent episodes to load

        Returns:
            List of episode dictionaries
        """
        return self.load_all_episodes(max_episodes=n_episodes)

    def create_training_batch(
        self,
        batch_size: int = 32,
        episodes: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Create a training batch from stored episodes.

        Args:
            batch_size: Number of transitions in batch
            episodes: List of episodes (if None, loads all)

        Returns:
            Dictionary with batch tensors
        """
        if episodes is None:
            episodes = self.load_all_episodes()

        if not episodes:
            raise ValueError("No episodes available for training")

        # Collect all transitions
        all_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs = []
        all_dones = []

        for episode in episodes:
            obs = episode['observations']
            actions = episode['actions']
            rewards = episode['rewards']
            dones = episode['dones']

            for t in range(len(obs) - 1):
                all_obs.append(obs[t])
                all_actions.append(actions[t])
                all_rewards.append(rewards[t])
                all_next_obs.append(obs[t + 1])
                all_dones.append(dones[t])

        # Sample random batch
        if len(all_obs) < batch_size:
            batch_size = len(all_obs)

        indices = np.random.choice(len(all_obs), batch_size, replace=False)

        # Create batch tensors
        batch = {
            'observations': torch.FloatTensor(np.array([all_obs[i] for i in indices])),
            'actions': torch.FloatTensor(np.array([all_actions[i] for i in indices])),
            'rewards': torch.FloatTensor(np.array([all_rewards[i] for i in indices])),
            'next_observations': torch.FloatTensor(np.array([all_next_obs[i] for i in indices])),
            'dones': torch.FloatTensor(np.array([all_dones[i] for i in indices])),
        }

        return batch

    def get_statistics(self) -> Dict:
        """Get statistics about stored training data"""
        return {
            'total_episodes': self.metadata['total_episodes'],
            'total_transitions': self.metadata['total_transitions'],
            'created_at': self.metadata['created_at'],
            'last_update': self.metadata['training_sessions'][-1]['timestamp'] if self.metadata['training_sessions'] else None,
            'sources': list(set(s['source'] for s in self.metadata['training_sessions'])),
        }

    def clear_old_data(self, keep_latest_n: int = 1000):
        """
        Clear old episodes, keeping only the latest N.

        Args:
            keep_latest_n: Number of latest episodes to keep
        """
        episode_files = sorted([
            f for f in os.listdir(self.episodes_dir)
            if f.startswith('episode_') and f.endswith('.pkl')
        ])

        if len(episode_files) > keep_latest_n:
            files_to_delete = episode_files[:-keep_latest_n]
            for file in files_to_delete:
                os.remove(os.path.join(self.episodes_dir, file))
            print(f"[*] Cleared {len(files_to_delete)} old episodes")


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = SharedTrainingDataManager()

    # Example: Save an episode
    observations = [np.random.randn(376) for _ in range(100)]
    actions = [np.random.randn(17) for _ in range(100)]
    rewards = [np.random.randn() for _ in range(100)]
    dones = [False] * 99 + [True]
    infos = [{}] * 100

    episode_id = manager.save_episode(
        observations, actions, rewards, dones, infos,
        source="ExampleScript"
    )
    print(f"[*] Saved episode: {episode_id}")

    # Example: Load all episodes
    episodes = manager.load_all_episodes()
    print(f"[*] Loaded {len(episodes)} episodes")

    # Example: Create training batch
    if episodes:
        batch = manager.create_training_batch(batch_size=32, episodes=episodes)
        print(f"[*] Created batch with {batch['observations'].shape[0]} transitions")

    # Example: Get statistics
    stats = manager.get_statistics()
    print(f"[*] Statistics: {stats}")
