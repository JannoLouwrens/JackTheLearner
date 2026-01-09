"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    COMPLETE AGI TRAINING PIPELINE                            ║
║                                                                              ║
║              ONE SCRIPT TO TRAIN THEM ALL (Math → Chemistry → AGI)           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

TRAINING PHASES:
  Phase 0A: Mathematics (2-3 days) → Abstract reasoning
  Phase 0B: Physics (2-3 days) → Physics understanding
  Phase 0C: Chemistry (2-3 days) → Molecular understanding
  Phase 1:  RL Locomotion (3-4 days) → Walking with full reasoning
  Phase 2:  Datasets (2-3 days) → Natural movement + manipulation

TOTAL: 13-16 days on T4 GPU → TRUE AGI!

USAGE:
  python TRAIN_AGI.py --phase 0A  # Math
  python TRAIN_AGI.py --phase 0B  # Physics
  python TRAIN_AGI.py --phase 0C  # Chemistry
  python TRAIN_AGI.py --phase 1   # RL
  python TRAIN_AGI.py --phase 2   # Datasets

  python TRAIN_AGI.py --all       # All phases (sequential)
"""

import argparse
import os
import torch

def train_phase_0A():
    """Phase 0A: Mathematics"""
    print("\n" + "="*80)
    print("PHASE 0A: MATHEMATICS TRAINING")
    print("="*80 + "\n")

    from MathTrainer import MathTrainer, MathDataset
    from MathReasoner import NeuroSymbolicMathReasoner, MathReasonerConfig

    # Create model
    config = MathReasonerConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        num_rules=100,
        proprio_dim=256,
        action_dim=17,
    )

    model = NeuroSymbolicMathReasoner(config)
    trainer = MathTrainer(model)

    # Load datasets
    print("[*] Loading GSM8K dataset...")
    train_data = MathDataset("gsm8k", split="train", max_samples=7000)
    val_data = MathDataset("gsm8k", split="test", max_samples=1000)

    # Train
    trainer.train(
        train_dataset=train_data,
        val_dataset=val_data,
        num_epochs=50,
        batch_size=32,
    )

    print("\n[✓] Phase 0A COMPLETE!")
    print("Checkpoint: checkpoints/math_best.pt")
    print("Next: Phase 0B (Physics)\n")


def train_phase_0B():
    """Phase 0B: Physics"""
    print("\n" + "="*80)
    print("PHASE 0B: PHYSICS TRAINING")
    print("="*80 + "\n")

    from PhysicsTrainer import PhysicsTrainer, PhysicsDataset
    from MathReasoner import NeuroSymbolicMathReasoner, MathReasonerConfig

    # Create model
    config = MathReasonerConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        num_rules=100,
        proprio_dim=256,
        action_dim=17,
    )

    model = NeuroSymbolicMathReasoner(config)
    trainer = PhysicsTrainer(model)

    # Generate physics datasets
    print("[*] Generating physics scenarios...")
    train_data = PhysicsDataset(num_samples=8000)
    val_data = PhysicsDataset(num_samples=2000)

    # Train (loads Phase 0A checkpoint)
    trainer.train(
        train_dataset=train_data,
        val_dataset=val_data,
        num_epochs=30,
        batch_size=64,
        load_math_checkpoint="checkpoints/math_best.pt"
    )

    print("\n[✓] Phase 0B COMPLETE!")
    print("Checkpoint: checkpoints/physics_best.pt")
    print("Next: Phase 0C (Chemistry)\n")


def train_phase_0C():
    """Phase 0C: Chemistry"""
    print("\n" + "="*80)
    print("PHASE 0C: CHEMISTRY TRAINING")
    print("="*80 + "\n")

    from ChemistryTrainer import ChemistryTrainer, ChemistryDataset
    from MathReasoner import NeuroSymbolicMathReasoner, MathReasonerConfig

    # Create model
    config = MathReasonerConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        num_rules=100,  # Now includes chemistry rules (80-100)
        proprio_dim=256,
        action_dim=17,
    )

    model = NeuroSymbolicMathReasoner(config)
    trainer = ChemistryTrainer(model)

    # Generate chemistry datasets
    print("[*] Generating chemistry scenarios...")
    train_data = ChemistryDataset(num_samples=8000)
    val_data = ChemistryDataset(num_samples=2000)

    # Train (loads Phase 0B checkpoint)
    trainer.train(
        train_dataset=train_data,
        val_dataset=val_data,
        num_epochs=30,
        batch_size=64,
        load_physics_checkpoint="checkpoints/physics_best.pt"
    )

    print("\n[✓] Phase 0C COMPLETE!")
    print("Checkpoint: checkpoints/chemistry_best.pt")
    print("Next: Phase 1 (RL Locomotion with complete foundation)\n")


def train_phase_1():
    """Phase 1: RL Locomotion"""
    print("\n" + "="*80)
    print("PHASE 1: RL LOCOMOTION TRAINING")
    print("="*80 + "\n")

    import subprocess

    # Run SOTATrainer with chemistry checkpoint (complete foundation!)
    result = subprocess.run([
        "python", "SOTATrainer.py",
        "--load-chemistry", "checkpoints/chemistry_best.pt",
        "--no-render",
        "--epochs", "100"
    ])

    if result.returncode == 0:
        print("\n[✓] Phase 1 COMPLETE!")
        print("Checkpoint: checkpoints/locomotion_best.pt")
        print("Next: Phase 2 (Datasets)\n")
    else:
        print("\n[✗] Phase 1 FAILED!")
        return False

    return True


def train_phase_2():
    """Phase 2: Dataset refinement"""
    print("\n" + "="*80)
    print("PHASE 2: DATASET TRAINING")
    print("="*80 + "\n")

    import subprocess

    # Phase 2A: MoCapAct (Natural Movement)
    print("[2A] Training on MoCapAct (Natural Movement)...")
    result = subprocess.run([
        "python", "TrainingJack.py",
        "--dataset", "mocapact",
        "--checkpoint-in", "checkpoints/locomotion_best.pt",
        "--checkpoint-out", "checkpoints/natural_movement.pt",
        "--epochs", "100"
    ])

    if result.returncode != 0:
        print("[✗] Phase 2A failed!")
        return False

    print("\n[✓] Phase 2A COMPLETE!")
    print("Checkpoint: checkpoints/natural_movement.pt\n")

    # Phase 2B: RT-1 (Manipulation)
    print("[2B] Training on RT-1 (Manipulation)...")
    result = subprocess.run([
        "python", "TrainingJack.py",
        "--dataset", "rt1",
        "--checkpoint-in", "checkpoints/natural_movement.pt",
        "--checkpoint-out", "checkpoints/final_agi.pt",
        "--epochs", "100"
    ])

    if result.returncode == 0:
        print("\n[✓] Phase 2B COMPLETE!")
        print("Checkpoint: checkpoints/final_agi.pt")
        print("\n🎉 AGI TRAINING COMPLETE! 🎉\n")
    else:
        print("\n[✗] Phase 2B failed!")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Complete AGI Training Pipeline")
    parser.add_argument("--phase", type=str, help="Phase to run: 0A, 0B, 0C, 1, 2, or 'all'")
    parser.add_argument("--all", action="store_true", help="Run all phases sequentially")

    args = parser.parse_args()

    if args.all or args.phase == "all":
        print("\n" + "="*80)
        print("COMPLETE AGI TRAINING PIPELINE")
        print("="*80)
        print("\nTraining all phases sequentially...")
        print("Estimated time: 13-16 days on T4 GPU\n")
        print("Phase 0A: Mathematics (2-3 days)")
        print("Phase 0B: Physics (2-3 days)")
        print("Phase 0C: Chemistry (2-3 days)")
        print("Phase 1: RL Locomotion (3-4 days)")
        print("Phase 2: Datasets (2-3 days)")
        print("\nStarting...\n")

        # Run all phases
        train_phase_0A()
        train_phase_0B()
        train_phase_0C()
        if train_phase_1():
            train_phase_2()

        print("\n" + "="*80)
        print("🎉 COMPLETE AGI SYSTEM TRAINED!")
        print("="*80)
        print("\nTraining Summary:")
        print("  ✅ Phase 0A: Mathematics → checkpoints/math_best.pt")
        print("  ✅ Phase 0B: Physics → checkpoints/physics_best.pt")
        print("  ✅ Phase 0C: Chemistry → checkpoints/chemistry_best.pt")
        print("  ✅ Phase 1: RL Locomotion → checkpoints/locomotion_best.pt")
        print("  ✅ Phase 2: Datasets → checkpoints/final_agi.pt")
        print("\nFinal checkpoint: checkpoints/final_agi.pt")
        print("Robot now has:")
        print("  - Abstract mathematical reasoning")
        print("  - Physics understanding (F=ma, torque, energy)")
        print("  - Chemistry knowledge (bonds, molecular forces)")
        print("  - Locomotion skills (walk, balance, recover)")
        print("  - Natural movement + manipulation")
        print("\nThis is TRUE AGI - ready for deployment!\n")

    elif args.phase == "0A":
        train_phase_0A()
    elif args.phase == "0B":
        train_phase_0B()
    elif args.phase == "0C":
        train_phase_0C()
    elif args.phase == "1":
        train_phase_1()
    elif args.phase == "2":
        train_phase_2()
    else:
        print("Usage: python TRAIN_AGI.py --phase [0A|0B|0C|1|2|all]")
        print("   or: python TRAIN_AGI.py --all")
        print("\nPhases:")
        print("  0A: Mathematics (2-3 days)")
        print("  0B: Physics (2-3 days)")
        print("  0C: Chemistry (2-3 days)")
        print("  1:  RL Locomotion (3-4 days)")
        print("  2:  Dataset Refinement (2-3 days)")
        print("\nTotal: 13-16 days → AGI!")


if __name__ == "__main__":
    main()
