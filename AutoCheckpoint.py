"""
AUTOMATIC CHECKPOINT MANAGER
Automatically finds and loads the latest/best checkpoint across all training scripts.

Both ProgressiveLearning and TrainingJack will use this to:
- Always continue from the most recent training
- Never lose progress
- Automatically build on previous training
"""

import os
import torch
from typing import Optional, Dict, Tuple
from datetime import datetime


class AutoCheckpointManager:
    """
    Automatically manages checkpoints across different training scripts.

    Features:
    - Finds the most recent checkpoint automatically
    - Loads best-performing checkpoint by default
    - Supports multiple checkpoint directories
    - No manual intervention needed
    """

    def __init__(self, checkpoint_dirs: list = ["checkpoints", "models"]):
        """
        Args:
            checkpoint_dirs: List of directories to search for checkpoints
        """
        self.checkpoint_dirs = checkpoint_dirs

    def find_all_checkpoints(self) -> Dict[str, Dict]:
        """
        Find all checkpoints across all directories.

        Returns:
            Dictionary mapping checkpoint paths to their metadata
        """
        all_checkpoints = {}

        for checkpoint_dir in self.checkpoint_dirs:
            if not os.path.exists(checkpoint_dir):
                continue

            # Find all .pt files
            for filename in os.listdir(checkpoint_dir):
                if not filename.endswith('.pt'):
                    continue

                filepath = os.path.join(checkpoint_dir, filename)

                try:
                    # Load checkpoint metadata (weights_only=False for full checkpoint)
                    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

                    # Extract metadata
                    metadata = {
                        'path': filepath,
                        'filename': filename,
                        'timestamp': checkpoint.get('timestamp', None),
                        'episode_count': checkpoint.get('episode_count', 0),
                        'epoch': checkpoint.get('epoch', 0),
                        'best_reward': checkpoint.get('best_reward', float('-inf')),
                        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
                        'current_stage': checkpoint.get('current_stage', 'unknown'),
                        'file_modified': datetime.fromtimestamp(os.path.getmtime(filepath)),
                    }

                    all_checkpoints[filepath] = metadata

                except Exception as e:
                    print(f"[WARNING] Could not load checkpoint {filepath}: {e}")
                    continue

        return all_checkpoints

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the most recently modified checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        all_checkpoints = self.find_all_checkpoints()

        if not all_checkpoints:
            return None

        # Sort by file modification time
        latest = max(all_checkpoints.items(), key=lambda x: x[1]['file_modified'])

        return latest[0]

    def get_best_checkpoint(self, prefer_progressive: bool = True) -> Optional[str]:
        """
        Get the best-performing checkpoint.

        Args:
            prefer_progressive: If True, prefer ProgressiveLearning checkpoints over TrainingJack

        Returns:
            Path to best checkpoint, or None if no checkpoints found
        """
        all_checkpoints = self.find_all_checkpoints()

        if not all_checkpoints:
            return None

        # Separate ProgressiveLearning and TrainingJack checkpoints
        progressive_ckpts = {
            path: meta for path, meta in all_checkpoints.items()
            if 'best_' in meta['filename'] or meta['episode_count'] > 0
        }

        training_jack_ckpts = {
            path: meta for path, meta in all_checkpoints.items()
            if 'models/' in path
        }

        # If prefer_progressive and we have progressive checkpoints, use those
        if prefer_progressive and progressive_ckpts:
            # Get best by reward (for ProgressiveLearning)
            best = max(progressive_ckpts.items(),
                      key=lambda x: x[1]['best_reward'])
            return best[0]

        # Otherwise, find best overall
        if training_jack_ckpts:
            # For TrainingJack, use best_val_loss
            best = min(training_jack_ckpts.items(),
                      key=lambda x: x[1]['best_val_loss'])
            return best[0]

        # Fallback to progressive checkpoints
        if progressive_ckpts:
            best = max(progressive_ckpts.items(),
                      key=lambda x: x[1]['best_reward'])
            return best[0]

        # Last resort: just get the latest
        return self.get_latest_checkpoint()

    def load_latest_checkpoint(
        self,
        brain,
        optimizer=None,
        device='cpu',
        prefer_best: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Automatically load the latest/best checkpoint.

        Args:
            brain: The robot brain model
            optimizer: Optional optimizer to restore
            device: Device to load checkpoint to
            prefer_best: If True, load best checkpoint. If False, load latest.

        Returns:
            (success, metadata) tuple
        """
        # Find checkpoint
        if prefer_best:
            checkpoint_path = self.get_best_checkpoint()
            checkpoint_type = "best"
        else:
            checkpoint_path = self.get_latest_checkpoint()
            checkpoint_type = "latest"

        if checkpoint_path is None:
            print("[*] No checkpoints found - starting fresh")
            return False, {}

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Load brain weights
            brain.load_state_dict(checkpoint['brain_state_dict'])

            # Load optimizer if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Print info
            print(f"[*] Auto-loaded {checkpoint_type} checkpoint: {checkpoint_path}")
            if 'current_stage' in checkpoint:
                print(f"   Stage: {checkpoint['current_stage']}")
            if 'episode_count' in checkpoint:
                print(f"   Episodes: {checkpoint['episode_count']}")
            if 'best_reward' in checkpoint:
                print(f"   Best reward: {checkpoint['best_reward']:.2f}")
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")

            return True, checkpoint

        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint {checkpoint_path}: {e}")
            return False, {}

    def list_checkpoints(self, max_display: int = 10):
        """
        List all available checkpoints with their info.

        Args:
            max_display: Maximum number of checkpoints to display
        """
        all_checkpoints = self.find_all_checkpoints()

        if not all_checkpoints:
            print("[*] No checkpoints found")
            return

        # Sort by modification time (newest first)
        sorted_ckpts = sorted(
            all_checkpoints.items(),
            key=lambda x: x[1]['file_modified'],
            reverse=True
        )

        print(f"\n[*] Found {len(sorted_ckpts)} checkpoints:")
        print("="*80)

        for i, (path, meta) in enumerate(sorted_ckpts[:max_display]):
            print(f"{i+1}. {meta['filename']}")
            print(f"   Path: {path}")
            print(f"   Modified: {meta['file_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            if meta['episode_count'] > 0:
                print(f"   Episodes: {meta['episode_count']}")
            if meta['epoch'] > 0:
                print(f"   Epoch: {meta['epoch']}")
            if meta['best_reward'] > float('-inf'):
                print(f"   Best reward: {meta['best_reward']:.2f}")
            print()

        if len(sorted_ckpts) > max_display:
            print(f"... and {len(sorted_ckpts) - max_display} more")

        print("="*80)


# Convenience function
def auto_load_checkpoint(brain, optimizer=None, device='cpu', prefer_best=True):
    """
    Convenience function to automatically load the latest checkpoint.

    Usage:
        success, metadata = auto_load_checkpoint(brain, optimizer, device='cuda')
    """
    manager = AutoCheckpointManager()
    return manager.load_latest_checkpoint(brain, optimizer, device, prefer_best)


# Example usage
if __name__ == "__main__":
    manager = AutoCheckpointManager()

    # List all checkpoints
    manager.list_checkpoints()

    # Get best checkpoint
    best = manager.get_best_checkpoint()
    print(f"\nBest checkpoint: {best}")

    # Get latest checkpoint
    latest = manager.get_latest_checkpoint()
    print(f"Latest checkpoint: {latest}")
