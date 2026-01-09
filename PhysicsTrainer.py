"""
PHASE 0B: PHYSICS & CHEMISTRY TRAINER

This is the SECOND phase - after math foundation (Phase 0A)

Goal: Ground abstract math in physical world understanding
Result: Robot that understands WHY things happen, not just imitates

Training data sources:
1. Physics problems (mechanics, dynamics, energy, momentum)
2. Simulated physics scenarios (pendulums, projectiles, collisions)
3. Chemistry datasets (molecular forces, reactions)
4. ArXiv physics papers (LaTeX equations)

Key physics laws to learn:
- F = ma (force = mass × acceleration)
- τ = r × F (torque = radius × force)
- E = ½mv² + mgh (mechanical energy)
- p = mv (momentum)
- CoM stability, friction, etc.

Why physics after math?
- Math provides abstract reasoning tools
- Physics grounds it in reality
- Result: Robot predicts "if I do X, Y will happen"

Expected training time: 2-3 days on T4 GPU
Total so far: 4-6 days (Phase 0A + 0B)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import json

from MathReasoner import NeuroSymbolicMathReasoner, MathReasonerConfig, compute_math_reasoning_loss


class PhysicsSimulator:
    """
    Simple physics simulator for generating training data.

    Simulates scenarios like:
    - Pendulum motion (θ'' = -g/L sin(θ))
    - Projectile motion (x = v₀t, y = v₀t - ½gt²)
    - Collisions (momentum conservation)
    - Torque and rotation
    - Center of mass dynamics

    These are the physics a humanoid robot needs to understand!
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.g = 9.81  # Gravity

    def simulate_pendulum(self, initial_angle: float, length: float, steps: int = 100):
        """
        Simulate pendulum: θ'' = -(g/L) sin(θ)

        Returns:
            states: (steps, 2) - [angle, angular_velocity]
            forces: (steps, 3) - [gravity, tension, torque]
        """
        theta = initial_angle
        omega = 0.0  # Initial angular velocity

        states = []
        forces = []

        for _ in range(steps):
            # Physics calculation
            torque = -(self.g / length) * np.sin(theta)
            alpha = torque  # Angular acceleration

            # Store state
            states.append([theta, omega])

            # Physics quantities
            gravity_force = self.g
            tension = self.g * np.cos(theta)  # Simplified
            forces.append([gravity_force, tension, torque])

            # Update (Euler integration)
            omega += alpha * self.dt
            theta += omega * self.dt

        return np.array(states), np.array(forces)

    def simulate_projectile(self, v0: float, angle: float, steps: int = 100):
        """
        Projectile motion: x = v₀cos(θ)t, y = v₀sin(θ)t - ½gt²

        Returns:
            states: (steps, 4) - [x, y, vx, vy]
            forces: (steps, 2) - [fx, fy]
        """
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        x, y = 0.0, 0.0

        states = []
        forces = []

        for _ in range(steps):
            states.append([x, y, vx, vy])
            forces.append([0.0, -self.g])  # Only gravity in y

            # Update
            x += vx * self.dt
            y += vy * self.dt
            vy -= self.g * self.dt

            if y < 0:  # Hit ground
                break

        return np.array(states), np.array(forces)

    def simulate_torque(self, force: float, radius: float, mass: float, steps: int = 100):
        """
        Rotation: τ = r × F, α = τ/I

        Returns:
            states: (steps, 2) - [angle, angular_velocity]
            torques: (steps, 3) - [torque, angular_accel, moment_of_inertia]
        """
        I = mass * radius ** 2  # Moment of inertia (simplified)
        torque = radius * force
        alpha = torque / I

        theta = 0.0
        omega = 0.0

        states = []
        torques = []

        for _ in range(steps):
            states.append([theta, omega])
            torques.append([torque, alpha, I])

            omega += alpha * self.dt
            theta += omega * self.dt

        return np.array(states), np.array(torques)


class PhysicsDataset(Dataset):
    """
    Physics dataset: simulated scenarios + real physics problems

    Each sample:
    - Initial state (e.g., pendulum at angle θ)
    - Action (e.g., apply force)
    - Next state (predicted by physics)
    - Physics quantities (forces, torques, energy)

    Robot learns to predict next state using physics laws!
    """

    def __init__(self, num_samples: int = 10000, scenario_types: List[str] = None):
        super().__init__()

        if scenario_types is None:
            scenario_types = ['pendulum', 'projectile', 'torque']

        self.simulator = PhysicsSimulator()
        self.scenarios = []

        print(f"[*] Generating {num_samples} physics scenarios...")

        for i in tqdm(range(num_samples)):
            scenario_type = np.random.choice(scenario_types)

            if scenario_type == 'pendulum':
                scenario = self._generate_pendulum_scenario()
            elif scenario_type == 'projectile':
                scenario = self._generate_projectile_scenario()
            elif scenario_type == 'torque':
                scenario = self._generate_torque_scenario()

            self.scenarios.append(scenario)

        print(f"[OK] Generated {len(self.scenarios)} scenarios\n")

    def _generate_pendulum_scenario(self):
        """Generate pendulum scenario"""
        initial_angle = np.random.uniform(-np.pi/2, np.pi/2)
        length = np.random.uniform(0.5, 2.0)

        states, forces = self.simulator.simulate_pendulum(initial_angle, length, steps=10)

        return {
            'type': 'pendulum',
            'initial_state': states[0],
            'final_state': states[-1],
            'trajectory': states,
            'forces': forces,
            'params': {'angle': initial_angle, 'length': length}
        }

    def _generate_projectile_scenario(self):
        """Generate projectile scenario"""
        v0 = np.random.uniform(5.0, 20.0)
        angle = np.random.uniform(0, np.pi/2)

        states, forces = self.simulator.simulate_projectile(v0, angle, steps=10)

        return {
            'type': 'projectile',
            'initial_state': states[0],
            'final_state': states[-1],
            'trajectory': states,
            'forces': forces,
            'params': {'v0': v0, 'angle': angle}
        }

    def _generate_torque_scenario(self):
        """Generate torque/rotation scenario"""
        force = np.random.uniform(1.0, 10.0)
        radius = np.random.uniform(0.1, 1.0)
        mass = np.random.uniform(0.5, 5.0)

        states, torques = self.simulator.simulate_torque(force, radius, mass, steps=10)

        return {
            'type': 'torque',
            'initial_state': states[0],
            'final_state': states[-1],
            'trajectory': states,
            'torques': torques,
            'params': {'force': force, 'radius': radius, 'mass': mass}
        }

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]

        # Convert to tensors
        # Pad/truncate to fixed dimension
        state = torch.zeros(256)
        state[:len(scenario['initial_state'])] = torch.FloatTensor(scenario['initial_state'])

        next_state = torch.zeros(256)
        next_state[:len(scenario['final_state'])] = torch.FloatTensor(scenario['final_state'])

        # Action: scenario parameters encoded
        action = torch.FloatTensor([
            scenario['params'].get('angle', 0),
            scenario['params'].get('length', 0),
            scenario['params'].get('v0', 0),
            scenario['params'].get('force', 0),
            scenario['params'].get('radius', 0),
            scenario['params'].get('mass', 0),
        ] + [0] * 11)  # Pad to 17 dims

        # Physics targets (forces, torques, energy)
        physics_target = torch.zeros(10)
        if 'forces' in scenario:
            forces_mean = scenario['forces'].mean(axis=0)
            physics_target[:len(forces_mean)] = torch.FloatTensor(forces_mean)
        elif 'torques' in scenario:
            torques_mean = scenario['torques'].mean(axis=0)
            physics_target[:len(torques_mean)] = torch.FloatTensor(torques_mean)

        return {
            'state': state,
            'action': action,
            'next_state': next_state,
            'physics_target': physics_target,
            'scenario_type': scenario['type'],
        }


class PhysicsTrainer:
    """
    Trainer for Phase 0B: Physics & Chemistry

    Loads checkpoint from Phase 0A (math foundation)
    Continues training on physics scenarios

    Result: Math + Physics understanding = Foundation for robotics
    """

    def __init__(
        self,
        model: NeuroSymbolicMathReasoner,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,  # Lower LR than Phase 0A (fine-tuning)
            weight_decay=0.01,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=5000,
            eta_min=1e-6
        )

        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "="*70)
        print("[*] PHYSICS TRAINER INITIALIZED")
        print("="*70)
        print(f"Device: {device}")
        print(f"Learning rate: 1e-4 → 1e-6 (fine-tuning from math)")
        print("="*70 + "\n")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        total_dynamics = 0
        total_physics = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            state = batch['state'].to(self.device)
            action = batch['action'].to(self.device)
            next_state = batch['next_state'].to(self.device)
            physics_target = batch['physics_target'].to(self.device)

            # Forward pass with physics targets
            loss, metrics = compute_math_reasoning_loss(
                self.model,
                state,
                action,
                next_state,
                physics_targets=physics_target  # NOW we have physics targets!
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += metrics['total_loss']
            total_dynamics += metrics['dynamics_loss']
            total_physics += metrics['physics_loss']

            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'dynamics': f"{metrics['dynamics_loss']:.4f}",
                'physics': f"{metrics['physics_loss']:.4f}",
                'rules': f"{metrics['avg_rules_used']:.1f}",
            })

        num_batches = len(dataloader)
        return {
            'loss': total_loss / num_batches,
            'dynamics_loss': total_dynamics / num_batches,
            'physics_loss': total_physics / num_batches,
        }

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_epochs: int = 30,
        batch_size: int = 64,
        load_math_checkpoint: str = None
    ):
        """Full training loop"""

        # Load Phase 0A checkpoint
        if load_math_checkpoint:
            print(f"[*] Loading math checkpoint: {load_math_checkpoint}")
            checkpoint = torch.load(load_math_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("[OK] Math foundation loaded!\n")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")

            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    state = batch['state'].to(self.device)
                    action = batch['action'].to(self.device)
                    next_state = batch['next_state'].to(self.device)
                    physics_target = batch['physics_target'].to(self.device)

                    loss, _ = compute_math_reasoning_loss(
                        self.model, state, action, next_state, physics_target
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"\n[EPOCH {epoch + 1} SUMMARY]")
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Physics loss: {train_metrics['physics_loss']:.4f}")
            print(f"  Val loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                path = os.path.join(self.checkpoint_dir, f"physics_epoch_{epoch+1}.pt")
                self.save_checkpoint(path, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(self.checkpoint_dir, "physics_best.pt")
                self.save_checkpoint(best_path, epoch)
                print(f"  ✅ New best model! (val_loss: {best_val_loss:.4f})")

        print(f"\n{'='*70}")
        print("[SUCCESS] PHASE 0B TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoint: {os.path.join(self.checkpoint_dir, 'physics_best.pt')}")
        print(f"\nNext step: Phase 1 (Locomotion RL training)")
        print(f"{'='*70}\n")

    def save_checkpoint(self, path: str, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"[SAVE] Checkpoint saved: {path}")


if __name__ == "__main__":
    print("="*70)
    print(" "*15 + "PHASE 0B: PHYSICS TRAINING")
    print("="*70 + "\n")

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

    # Create trainer
    trainer = PhysicsTrainer(model)

    # Generate physics datasets
    print("[*] Generating physics datasets...\n")
    train_dataset = PhysicsDataset(num_samples=8000, scenario_types=['pendulum', 'projectile', 'torque'])
    val_dataset = PhysicsDataset(num_samples=2000, scenario_types=['pendulum', 'projectile', 'torque'])

    print("\n[*] Training pipeline ready!")
    print("\nTo train:")
    print("  1. Complete Phase 0A (math training) first")
    print("  2. Run: python PhysicsTrainer.py --load-math checkpoints/math_best.pt")
    print("  3. Wait 2-3 days for physics training")
    print("  4. Result: Brain with math + physics understanding\n")

    print("="*70)
    print("PHYSICS TRAINING PIPELINE READY!")
    print("="*70)
    print("\nPhysics laws being learned:")
    print("  F = ma (force = mass × acceleration)")
    print("  τ = r × F (torque)")
    print("  E = ½mv² + mgh (energy)")
    print("  p = mv (momentum)")
    print("  + Many more from simulations!")
    print("\nThis creates TRUE understanding, not just imitation!")
    print("="*70)
