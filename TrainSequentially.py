"""
SEQUENTIAL TRAINING SCRIPT
Train Jack's brain in the correct order for optimal results:

1. Basic RL locomotion (ProgressiveLearning)
2. Natural human movement (CMU MoCap)
3. Refined robot movement (MoCapAct)
4. Physics-aware movement (DeepMind)
5. Object manipulation (RT-1)
6. Language understanding (Language-Table)
"""

import os
import sys
from typing import Optional

class SequentialTrainer:
    """
    Manages sequential training across different datasets.
    Each stage builds on the previous one.
    """

    def __init__(self):
        self.stages = [
            {
                "name": "Phase 1: Basic Locomotion (RL)",
                "script": "ProgressiveLearning.py",
                "checkpoint_out": "checkpoints/locomotion.pt",
                "description": "Learn to stand and walk using reinforcement learning",
                "dataset": None,
                "estimated_time": "1-2 weeks",
                "priority": 0,
            },
            {
                "name": "Phase 2A: Natural Human Movement",
                "script": "TrainingJack.py",
                "checkpoint_in": "checkpoints/locomotion.pt",
                "checkpoint_out": "checkpoints/natural_movement.pt",
                "description": "Learn natural human movements from motion capture",
                "dataset": "cmu_mocap",
                "estimated_time": "4-8 hours",
                "priority": 1,
            },
            {
                "name": "Phase 2B: Refined Robot Movement",
                "script": "TrainingJack.py",
                "checkpoint_in": "checkpoints/natural_movement.pt",
                "checkpoint_out": "checkpoints/refined_movement.pt",
                "description": "Refine movements for robot constraints",
                "dataset": "mocapact",
                "estimated_time": "6-12 hours",
                "priority": 2,
            },
            {
                "name": "Phase 2C: Physics-Aware Movement",
                "script": "TrainingJack.py",
                "checkpoint_in": "checkpoints/refined_movement.pt",
                "checkpoint_out": "checkpoints/physics_movement.pt",
                "description": "Learn physics-aware locomotion",
                "dataset": "deepmind_control",
                "estimated_time": "6-12 hours",
                "priority": 3,
            },
            {
                "name": "Phase 2D: Object Manipulation",
                "script": "TrainingJack.py",
                "checkpoint_in": "checkpoints/physics_movement.pt",
                "checkpoint_out": "checkpoints/manipulation.pt",
                "description": "Learn to manipulate objects",
                "dataset": "rt1_subset",
                "estimated_time": "12-24 hours",
                "priority": 4,
            },
            {
                "name": "Phase 2E: Language Understanding",
                "script": "TrainingJack.py",
                "checkpoint_in": "checkpoints/manipulation.pt",
                "checkpoint_out": "checkpoints/multimodal.pt",
                "description": "Learn language-conditioned actions",
                "dataset": "language_table",
                "estimated_time": "8-16 hours",
                "priority": 5,
            },
        ]

        self.current_stage = 0

    def check_status(self):
        """Check which stages are completed"""
        print("\n" + "="*70)
        print("SEQUENTIAL TRAINING STATUS")
        print("="*70)

        for i, stage in enumerate(self.stages):
            checkpoint = stage.get('checkpoint_out')
            dataset = stage.get('dataset')

            # Check if completed
            if checkpoint and os.path.exists(checkpoint):
                status = "[COMPLETE]"
                self.current_stage = i + 1
            elif i == self.current_stage:
                status = "[CURRENT]"
            else:
                status = "[PENDING]"

            print(f"\n{status} Stage {i+1}: {stage['name']}")
            print(f"   Description: {stage['description']}")
            print(f"   Time: {stage['estimated_time']}")

            if dataset:
                dataset_path = f"datasets/{self.get_dataset_dir(dataset)}"
                if os.path.exists(dataset_path) and os.listdir(dataset_path):
                    print(f"   Dataset: {dataset} [OK]")
                else:
                    print(f"   Dataset: {dataset} [MISSING - Download needed]")

            if checkpoint:
                if os.path.exists(checkpoint):
                    size_mb = os.path.getsize(checkpoint) / (1024*1024)
                    print(f"   Checkpoint: {checkpoint} ({size_mb:.1f}MB)")
                else:
                    print(f"   Checkpoint: {checkpoint} [Not created yet]")

        print("\n" + "="*70)
        print(f"Current Stage: {self.current_stage + 1}/{len(self.stages)}")
        print("="*70)

    def get_dataset_dir(self, dataset_key: str) -> str:
        """Get dataset directory name"""
        mapping = {
            "cmu_mocap": "cmu_mocap",
            "mocapact": "mocapact",
            "deepmind_control": "deepmind_mocap",
            "rt1_subset": "rt1",
            "language_table": "language_table",
        }
        return mapping.get(dataset_key, dataset_key)

    def get_next_stage(self) -> Optional[dict]:
        """Get next stage to train"""
        if self.current_stage >= len(self.stages):
            return None
        return self.stages[self.current_stage]

    def show_next_action(self):
        """Show what to do next"""
        next_stage = self.get_next_stage()

        if next_stage is None:
            print("\n" + "="*70)
            print("ALL STAGES COMPLETE!")
            print("="*70)
            print("\nYour robot has completed all training phases!")
            print("Next: Deploy to real hardware (Phase 3)")
            print("\nFinal checkpoint: checkpoints/multimodal.pt")
            print("="*70)
            return

        print("\n" + "="*70)
        print("NEXT ACTION")
        print("="*70)
        print(f"\nStage {self.current_stage + 1}: {next_stage['name']}")
        print(f"Description: {next_stage['description']}")
        print(f"Estimated time: {next_stage['estimated_time']}")

        # Check prerequisites
        dataset = next_stage.get('dataset')
        checkpoint_in = next_stage.get('checkpoint_in')

        print("\n[*] Prerequisites:")

        # Check dataset
        if dataset:
            dataset_path = f"datasets/{self.get_dataset_dir(dataset)}"
            if os.path.exists(dataset_path) and os.listdir(dataset_path):
                print(f"   [OK] Dataset '{dataset}' is ready")
            else:
                print(f"   [MISSING] Dataset '{dataset}' needs to be downloaded")
                print(f"   Action: py DatasetDownloader.py --download {dataset}")
                print("")
                return

        # Check input checkpoint
        if checkpoint_in:
            if os.path.exists(checkpoint_in):
                print(f"   [OK] Input checkpoint '{checkpoint_in}' exists")
            else:
                print(f"   [MISSING] Input checkpoint '{checkpoint_in}' not found")
                print(f"   Action: Complete previous stage first")
                print("")
                return

        # Ready to train!
        print("\n[*] All prerequisites met! Ready to train.")
        print(f"\n[*] To start training:")
        if next_stage['script'] == "ProgressiveLearning.py":
            print(f"   py {next_stage['script']} --no-render")
        else:
            print(f"   py {next_stage['script']} --dataset {dataset}")
            if checkpoint_in:
                print(f"   (Will auto-load checkpoint: {checkpoint_in})")

        print("\n" + "="*70)

    def show_full_plan(self):
        """Show the complete training plan"""
        print("\n" + "="*70)
        print("COMPLETE TRAINING PLAN")
        print("="*70)
        print("\nSequential training from basic to complex:")

        total_time_min = 0
        total_time_max = 0

        for i, stage in enumerate(self.stages):
            print(f"\n{i+1}. {stage['name']}")
            print(f"   {stage['description']}")
            print(f"   Time: {stage['estimated_time']}")

            if stage.get('dataset'):
                print(f"   Dataset: {stage['dataset']}")

            if stage.get('checkpoint_in'):
                print(f"   Input: {stage['checkpoint_in']}")
            print(f"   Output: {stage['checkpoint_out']}")

            # Parse time estimate
            time_str = stage['estimated_time']
            if 'week' in time_str:
                min_days = int(time_str.split('-')[0]) * 7
                max_days = int(time_str.split('-')[1].split()[0]) * 7
                total_time_min += min_days * 24
                total_time_max += max_days * 24
            elif 'hours' in time_str:
                min_hrs = int(time_str.split('-')[0])
                max_hrs = int(time_str.split('-')[1].split()[0])
                total_time_min += min_hrs
                total_time_max += max_hrs

        print("\n" + "="*70)
        print(f"TOTAL TIME ESTIMATE: {total_time_min}-{total_time_max} hours")
        print(f"                     (~{total_time_min//24}-{total_time_max//24} days)")
        print("="*70)
        print("\nThis produces a fully capable robot with:")
        print("- Natural human-like movement")
        print("- Object manipulation skills")
        print("- Vision understanding")
        print("- Language comprehension")
        print("- Ready for real-world deployment")
        print("="*70)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Sequential training manager")
    parser.add_argument("--status", action="store_true", help="Show training status")
    parser.add_argument("--next", action="store_true", help="Show next action")
    parser.add_argument("--plan", action="store_true", help="Show complete training plan")
    args = parser.parse_args()

    trainer = SequentialTrainer()

    if args.status:
        trainer.check_status()
    elif args.next:
        trainer.check_status()
        trainer.show_next_action()
    elif args.plan:
        trainer.show_full_plan()
    else:
        # Default: show status and next action
        trainer.check_status()
        trainer.show_next_action()


if __name__ == "__main__":
    main()
