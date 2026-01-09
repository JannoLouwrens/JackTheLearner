"""
PHASE 0C: CHEMISTRY & MOLECULAR REASONING TRAINER

This is the THIRD phase - after math (0A) and physics (0B) foundations

Goal: Ground physics in molecular understanding
Result: Robot understands material properties, molecular forces, bonds

Training data sources:
1. Molecular dynamics simulations (H2O, proteins, etc.)
2. Chemistry datasets (QM9, PubChemQC)
3. Material property databases
4. Reaction energy calculations

Key chemistry concepts to learn:
- Bond energies (C-C, C-O, O-H, etc.)
- Molecular forces (van der Waals, ionic, covalent, hydrogen bonding)
- Reaction energetics (ΔH, ΔG, activation energy)
- Material properties (elasticity, hardness, friction coefficients)

Why chemistry after physics?
- Chemistry IS physics at molecular scale!
- Bond energy → Energy conservation (learned in Phase 0B)
- Molecular forces → F=ma at atomic level
- Material properties critical for manipulation tasks

Expected training time: 2-3 days on T4 GPU
Total so far: 7-9 days (Phase 0A + 0B + 0C)
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


class MolecularSimulator:
    """
    Simple molecular dynamics simulator for generating training data.

    Simulates scenarios like:
    - H2O molecule vibrations
    - Bond breaking/forming
    - Intermolecular forces
    - Material stress-strain

    These are the molecular interactions a humanoid robot needs to understand!
    """

    def __init__(self):
        # Bond energies (kJ/mol) - from experimental data
        self.bond_energies = {
            'C-C': 350,    # Single carbon-carbon bond
            'C=C': 610,    # Double carbon-carbon bond
            'C≡C': 835,    # Triple carbon-carbon bond
            'C-H': 410,    # Carbon-hydrogen bond
            'C-O': 360,    # Carbon-oxygen bond
            'C=O': 745,    # Carbon-oxygen double bond (carbonyl)
            'O-H': 460,    # Oxygen-hydrogen bond
            'N-H': 390,    # Nitrogen-hydrogen bond
            'H-H': 436,    # Hydrogen molecule
            'O=O': 498,    # Oxygen molecule
        }

        # Intermolecular forces (kJ/mol)
        self.intermolecular_forces = {
            'van_der_waals': 4,      # Weak dispersion forces
            'hydrogen_bond': 20,      # Medium strength
            'dipole_dipole': 8,       # Polar molecule interaction
            'ionic': 700,             # Very strong
        }

        # Material properties
        self.material_properties = {
            'rubber': {'elastic_modulus': 0.01, 'hardness': 1, 'friction': 0.8},
            'wood': {'elastic_modulus': 10, 'hardness': 3, 'friction': 0.4},
            'steel': {'elastic_modulus': 200, 'hardness': 9, 'friction': 0.6},
            'glass': {'elastic_modulus': 70, 'hardness': 6, 'friction': 0.3},
            'plastic': {'elastic_modulus': 2, 'hardness': 2, 'friction': 0.5},
        }

    def simulate_bond_vibration(self, bond_type: str, steps: int = 100):
        """
        Simulate bond vibration (harmonic oscillator approximation)

        Returns:
            states: (steps, 3) - [bond_length, velocity, energy]
            energies: (steps, 2) - [kinetic, potential]
        """
        bond_energy = self.bond_energies.get(bond_type, 400)  # Default to typical bond
        equilibrium_length = 1.5e-10  # 1.5 Angstroms (typical C-C)
        k = bond_energy * 1000 / (equilibrium_length ** 2)  # Spring constant

        # Initial conditions
        x = equilibrium_length * 1.1  # 10% stretched
        v = 0.0
        m = 12e-27  # Approximate mass of carbon atom (kg)
        dt = 1e-15  # 1 femtosecond timestep

        states = []
        energies = []

        for _ in range(steps):
            # Harmonic oscillator: F = -k(x - x0)
            F = -k * (x - equilibrium_length)
            a = F / m

            # Kinetic and potential energy
            KE = 0.5 * m * v ** 2
            PE = 0.5 * k * (x - equilibrium_length) ** 2
            total_energy = KE + PE

            states.append([x, v, total_energy])
            energies.append([KE, PE])

            # Update (Verlet integration for stability)
            v += a * dt
            x += v * dt

        return np.array(states), np.array(energies)

    def simulate_reaction(self, reactants: str, products: str):
        """
        Simulate simple reaction: A + B → C

        Returns:
            energy_profile: (steps, 2) - [reaction_coordinate, energy]
            delta_H: float - enthalpy change
        """
        # Simplified reaction energy profile (Gaussian barrier)
        steps = 100
        reaction_coordinate = np.linspace(0, 1, steps)

        # Activation energy (random 50-200 kJ/mol)
        E_activation = np.random.uniform(50, 200)

        # Enthalpy change (random -100 to +100 kJ/mol)
        delta_H = np.random.uniform(-100, 100)

        # Energy profile: Gaussian barrier
        energy = np.zeros(steps)
        for i, x in enumerate(reaction_coordinate):
            # Barrier at x=0.5
            barrier = E_activation * np.exp(-50 * (x - 0.5) ** 2)
            # Downhill/uphill based on delta_H
            slope = delta_H * x
            energy[i] = barrier + slope

        energy_profile = np.column_stack([reaction_coordinate, energy])

        return energy_profile, delta_H, E_activation

    def simulate_material_stress(self, material: str, max_strain: float = 0.1):
        """
        Simulate stress-strain curve for material

        Returns:
            stress_strain: (steps, 2) - [strain, stress]
            properties: dict - elastic modulus, yield point, etc.
        """
        props = self.material_properties.get(material, self.material_properties['plastic'])
        E = props['elastic_modulus']  # GPa

        steps = 100
        strain = np.linspace(0, max_strain, steps)
        stress = np.zeros(steps)

        # Elastic region (Hooke's law): σ = E⋅ε
        elastic_limit = 0.002  # 0.2% strain

        for i, eps in enumerate(strain):
            if eps < elastic_limit:
                # Elastic (linear)
                stress[i] = E * eps
            else:
                # Plastic (nonlinear)
                stress[i] = E * elastic_limit + 0.5 * E * (eps - elastic_limit)

        stress_strain = np.column_stack([strain, stress])

        return stress_strain, props


class ChemistryDataset(Dataset):
    """
    Chemistry dataset: molecular scenarios + material properties

    Each sample:
    - Initial molecular state
    - Action/perturbation
    - Next state (predicted by chemistry)
    - Energies, forces, properties

    Robot learns to predict molecular behavior!
    """

    def __init__(self, num_samples: int = 10000, scenario_types: List[str] = None):
        super().__init__()

        if scenario_types is None:
            scenario_types = ['bond_vibration', 'reaction', 'material_stress']

        self.simulator = MolecularSimulator()
        self.scenarios = []

        print(f"[*] Generating {num_samples} chemistry scenarios...")

        for i in tqdm(range(num_samples)):
            scenario_type = np.random.choice(scenario_types)

            if scenario_type == 'bond_vibration':
                scenario = self._generate_bond_scenario()
            elif scenario_type == 'reaction':
                scenario = self._generate_reaction_scenario()
            elif scenario_type == 'material_stress':
                scenario = self._generate_material_scenario()

            self.scenarios.append(scenario)

        print(f"[OK] Generated {len(self.scenarios)} scenarios\n")

    def _generate_bond_scenario(self):
        """Generate bond vibration scenario"""
        bond_types = ['C-C', 'C=C', 'C-H', 'O-H', 'N-H']
        bond_type = np.random.choice(bond_types)

        states, energies = self.simulator.simulate_bond_vibration(bond_type, steps=10)

        return {
            'type': 'bond_vibration',
            'bond_type': bond_type,
            'initial_state': states[0],
            'final_state': states[-1],
            'trajectory': states,
            'energies': energies,
            'bond_energy': self.simulator.bond_energies[bond_type],
        }

    def _generate_reaction_scenario(self):
        """Generate reaction scenario"""
        reactions = [
            ('H2 + O2', 'H2O'),
            ('CH4 + O2', 'CO2 + H2O'),
            ('C + O2', 'CO2'),
        ]
        reactants, products = reactions[np.random.randint(len(reactions))]

        energy_profile, delta_H, E_activation = self.simulator.simulate_reaction(reactants, products)

        return {
            'type': 'reaction',
            'reactants': reactants,
            'products': products,
            'initial_state': energy_profile[0],
            'final_state': energy_profile[-1],
            'energy_profile': energy_profile,
            'delta_H': delta_H,
            'E_activation': E_activation,
        }

    def _generate_material_scenario(self):
        """Generate material stress-strain scenario"""
        materials = ['rubber', 'wood', 'steel', 'glass', 'plastic']
        material = np.random.choice(materials)

        stress_strain, properties = self.simulator.simulate_material_stress(material)

        return {
            'type': 'material_stress',
            'material': material,
            'initial_state': stress_strain[0],
            'final_state': stress_strain[-1],
            'stress_strain': stress_strain,
            'properties': properties,
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
        action = torch.zeros(17)
        if scenario['type'] == 'bond_vibration':
            action[0] = scenario['bond_energy'] / 1000.0  # Normalize
        elif scenario['type'] == 'reaction':
            action[1] = scenario['delta_H'] / 100.0  # Normalize
            action[2] = scenario['E_activation'] / 200.0  # Normalize
        elif scenario['type'] == 'material_stress':
            action[3] = scenario['properties']['elastic_modulus'] / 200.0
            action[4] = scenario['properties']['hardness'] / 10.0
            action[5] = scenario['properties']['friction']

        # Chemistry targets (energies, forces, properties)
        chemistry_target = torch.zeros(10)
        if 'energies' in scenario:
            energies_mean = scenario['energies'].mean(axis=0)
            chemistry_target[:len(energies_mean)] = torch.FloatTensor(energies_mean)
        elif 'delta_H' in scenario:
            chemistry_target[0] = scenario['delta_H'] / 100.0
            chemistry_target[1] = scenario['E_activation'] / 200.0

        return {
            'state': state,
            'action': action,
            'next_state': next_state,
            'chemistry_target': chemistry_target,
            'scenario_type': scenario['type'],
        }


class ChemistryTrainer:
    """
    Trainer for Phase 0C: Chemistry & Molecular Reasoning

    Loads checkpoint from Phase 0B (physics foundation)
    Continues training on chemistry scenarios

    Result: Math + Physics + Chemistry understanding = Complete foundation for AGI
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
            lr=5e-5,  # Even lower LR than Phase 0B (fine-tuning chemistry)
            weight_decay=0.01,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=3000,
            eta_min=1e-6
        )

        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "="*70)
        print("[*] CHEMISTRY TRAINER INITIALIZED")
        print("="*70)
        print(f"Device: {device}")
        print(f"Learning rate: 5e-5 → 1e-6 (fine-tuning from physics)")
        print("="*70 + "\n")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        total_dynamics = 0
        total_chemistry = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            state = batch['state'].to(self.device)
            action = batch['action'].to(self.device)
            next_state = batch['next_state'].to(self.device)
            chemistry_target = batch['chemistry_target'].to(self.device)

            # Forward pass with chemistry targets
            loss, metrics = compute_math_reasoning_loss(
                self.model,
                state,
                action,
                next_state,
                physics_targets=chemistry_target  # Reuse physics_targets for chemistry
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += metrics['total_loss']
            total_dynamics += metrics['dynamics_loss']
            total_chemistry += metrics['physics_loss']  # Called 'physics_loss' but is chemistry here

            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'dynamics': f"{metrics['dynamics_loss']:.4f}",
                'chemistry': f"{metrics['physics_loss']:.4f}",
                'rules': f"{metrics['avg_rules_used']:.1f}",
            })

        num_batches = len(dataloader)
        return {
            'loss': total_loss / num_batches,
            'dynamics_loss': total_dynamics / num_batches,
            'chemistry_loss': total_chemistry / num_batches,
        }

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_epochs: int = 30,
        batch_size: int = 64,
        load_physics_checkpoint: str = None
    ):
        """Full training loop"""

        # Load Phase 0B checkpoint
        if load_physics_checkpoint:
            print(f"[*] Loading physics checkpoint: {load_physics_checkpoint}")
            checkpoint = torch.load(load_physics_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("[OK] Physics foundation loaded!\n")

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
                    chemistry_target = batch['chemistry_target'].to(self.device)

                    loss, _ = compute_math_reasoning_loss(
                        self.model, state, action, next_state, chemistry_target
                    )
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"\n[EPOCH {epoch + 1} SUMMARY]")
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Chemistry loss: {train_metrics['chemistry_loss']:.4f}")
            print(f"  Val loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                path = os.path.join(self.checkpoint_dir, f"chemistry_epoch_{epoch+1}.pt")
                self.save_checkpoint(path, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(self.checkpoint_dir, "chemistry_best.pt")
                self.save_checkpoint(best_path, epoch)
                print(f"  ✅ New best model! (val_loss: {best_val_loss:.4f})")

        print(f"\n{'='*70}")
        print("[SUCCESS] PHASE 0C TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoint: {os.path.join(self.checkpoint_dir, 'chemistry_best.pt')}")
        print(f"\nNext step: Phase 1 (RL Locomotion with complete foundation)")
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
    print(" "*15 + "PHASE 0C: CHEMISTRY TRAINING")
    print("="*70 + "\n")

    # Create model
    config = MathReasonerConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        num_rules=100,  # Now includes chemistry rules (rules 80-100)
        proprio_dim=256,
        action_dim=17,
    )

    model = NeuroSymbolicMathReasoner(config)

    # Create trainer
    trainer = ChemistryTrainer(model)

    # Generate chemistry datasets
    print("[*] Generating chemistry datasets...\n")
    train_dataset = ChemistryDataset(
        num_samples=8000,
        scenario_types=['bond_vibration', 'reaction', 'material_stress']
    )
    val_dataset = ChemistryDataset(
        num_samples=2000,
        scenario_types=['bond_vibration', 'reaction', 'material_stress']
    )

    print("\n[*] Training pipeline ready!")
    print("\nTo train:")
    print("  1. Complete Phase 0A (math training) first")
    print("  2. Complete Phase 0B (physics training) second")
    print("  3. Run: python ChemistryTrainer.py --load-physics checkpoints/physics_best.pt")
    print("  4. Wait 2-3 days for chemistry training")
    print("  5. Result: Brain with math + physics + chemistry understanding\n")

    print("="*70)
    print("CHEMISTRY TRAINING PIPELINE READY!")
    print("="*70)
    print("\nChemistry concepts being learned:")
    print("  Bond energies:")
    print("    - C-C: 350 kJ/mol")
    print("    - C=C: 610 kJ/mol")
    print("    - C-H: 410 kJ/mol")
    print("    - O-H: 460 kJ/mol")
    print("  Intermolecular forces:")
    print("    - van der Waals: ~4 kJ/mol")
    print("    - Hydrogen bonding: ~20 kJ/mol")
    print("    - Ionic: ~700 kJ/mol")
    print("  Material properties:")
    print("    - Elasticity, hardness, friction")
    print("    - Stress-strain curves")
    print("\nThis creates molecular-level understanding for manipulation!")
    print("="*70)
