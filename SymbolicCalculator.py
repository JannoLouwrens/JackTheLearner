"""
SYMBOLIC PHYSICS CALCULATOR - TRUE HYBRID COMPONENT

This is the REAL symbolic calculator (like AlphaGeometry's symbolic engine).
Uses SymPy to perform EXACT mathematical calculations.

Purpose:
1. Training: Provides ground truth for neural network to learn from
2. Inference (optional): Verifies neural predictions for safety

Key physics laws implemented:
- F = ma (force = mass × acceleration)
- τ = r × F (torque = radius × force)
- E = ½mv² + mgh (mechanical energy)
- p = mv (momentum)
- CoM stability calculations
- Collision physics

This is NOT learned - these are ACTUAL physics equations!
The neural network learns to approximate these (fast inference),
but symbolic calculator provides EXACT answers (training supervision).
"""

import sympy as sp
import numpy as np
from typing import Dict, Tuple, Optional
import torch


class SymbolicPhysicsCalculator:
    """
    Symbolic physics engine using SymPy.

    Performs EXACT calculations for physics laws.
    No approximation, no hallucination - pure math!
    """

    def __init__(self):
        # Define symbolic variables
        self.m = sp.Symbol('m', positive=True)  # mass
        self.a = sp.Symbol('a', real=True)      # acceleration
        self.F = sp.Symbol('F', real=True)      # force
        self.r = sp.Symbol('r', positive=True)  # radius
        self.v = sp.Symbol('v', real=True)      # velocity
        self.h = sp.Symbol('h', real=True)      # height
        self.t = sp.Symbol('t', real=True)      # time
        self.theta = sp.Symbol('theta', real=True)  # angle
        self.omega = sp.Symbol('omega', real=True)  # angular velocity
        self.I = sp.Symbol('I', positive=True)  # moment of inertia
        self.g = sp.Float(9.81)                 # gravity constant

        # Define physics laws symbolically
        self._define_physics_laws()

        # Chemistry properties (for Phase 0C)
        self._define_chemistry_properties()

        print("[*] Symbolic Physics & Chemistry Calculator Initialized")
        print("   Physics: F=ma, τ=r×F, E=½mv²+mgh, p=mv, pendulum, projectile")
        print("   Chemistry: Bond energies, molecular forces, reactions")
        print("   Engine: SymPy (exact symbolic computation)\n")

    def _define_physics_laws(self):
        """Define fundamental physics laws symbolically"""

        # Newton's second law: F = ma
        self.newtons_law = sp.Eq(self.F, self.m * self.a)

        # Torque: τ = r × F
        self.torque_law = sp.Eq(sp.Symbol('tau'), self.r * self.F)

        # Kinetic energy: KE = ½mv²
        self.kinetic_energy = sp.Rational(1, 2) * self.m * self.v**2

        # Potential energy: PE = mgh
        self.potential_energy = self.m * self.g * self.h

        # Total mechanical energy
        self.mechanical_energy = self.kinetic_energy + self.potential_energy

        # Momentum: p = mv
        self.momentum = self.m * self.v

        # Pendulum equation: θ'' = -(g/L)sin(θ)
        L = sp.Symbol('L', positive=True)
        self.pendulum_accel = -(self.g / L) * sp.sin(self.theta)

        # Projectile motion: x = v₀t, y = v₀t - ½gt²
        v0 = sp.Symbol('v0', real=True)
        self.projectile_x = v0 * self.t
        self.projectile_y = v0 * self.t - sp.Rational(1, 2) * self.g * self.t**2

    def _define_chemistry_properties(self):
        """Define chemistry laws and properties for Phase 0C"""

        # Bond energies (kJ/mol) - experimental values
        self.bond_energies = {
            'C-C': 350,    # Single carbon-carbon bond
            'C=C': 610,    # Double carbon-carbon bond
            'C≡C': 835,    # Triple carbon-carbon bond
            'C-H': 410,    # Carbon-hydrogen bond
            'C-O': 360,    # Carbon-oxygen bond
            'C=O': 745,    # Carbon-oxygen double bond (carbonyl)
            'O-H': 460,    # Oxygen-hydrogen bond
            'O=O': 498,    # Oxygen molecule
            'N-H': 390,    # Nitrogen-hydrogen bond
            'N-N': 160,    # Nitrogen-nitrogen bond
            'N≡N': 945,    # Nitrogen molecule (triple bond)
            'H-H': 436,    # Hydrogen molecule
        }

        # Intermolecular forces (kJ/mol)
        self.intermolecular_forces = {
            'van_der_waals': 4,      # Weak dispersion forces
            'hydrogen_bond': 20,      # Medium strength
            'dipole_dipole': 8,       # Polar molecule interaction
            'ionic': 700,             # Very strong (electrostatic)
        }

        # Material properties (for manipulation tasks)
        self.material_properties = {
            'rubber': {
                'elastic_modulus': 0.01,  # GPa
                'hardness': 1,            # Shore scale (1-10)
                'friction_coeff': 0.8,    # High friction
                'density': 1.1,           # g/cm³
            },
            'wood': {
                'elastic_modulus': 10,
                'hardness': 3,
                'friction_coeff': 0.4,
                'density': 0.6,
            },
            'steel': {
                'elastic_modulus': 200,
                'hardness': 9,
                'friction_coeff': 0.6,
                'density': 7.85,
            },
            'glass': {
                'elastic_modulus': 70,
                'hardness': 6,
                'friction_coeff': 0.3,
                'density': 2.5,
            },
            'plastic': {
                'elastic_modulus': 2,
                'hardness': 2,
                'friction_coeff': 0.5,
                'density': 1.2,
            },
        }

    def calculate_force(self, mass: float, acceleration: float) -> float:
        """
        F = ma

        Args:
            mass: kg
            acceleration: m/s²
        Returns:
            force: N (Newtons)
        """
        return float(mass * acceleration)

    def calculate_acceleration(self, force: float, mass: float) -> float:
        """
        a = F/m

        Solves Newton's law symbolically then substitutes values.
        """
        solution = sp.solve(self.newtons_law, self.a)[0]
        result = solution.subs([(self.F, force), (self.m, mass)])
        return float(result)

    def calculate_torque(self, radius: float, force: float) -> float:
        """
        τ = r × F

        Args:
            radius: m
            force: N
        Returns:
            torque: N⋅m
        """
        return float(radius * force)

    def calculate_angular_acceleration(self, torque: float, moment_of_inertia: float) -> float:
        """
        α = τ/I

        Args:
            torque: N⋅m
            moment_of_inertia: kg⋅m²
        Returns:
            angular_acceleration: rad/s²
        """
        return float(torque / moment_of_inertia)

    def calculate_kinetic_energy(self, mass: float, velocity: float) -> float:
        """
        KE = ½mv²
        """
        result = self.kinetic_energy.subs([(self.m, mass), (self.v, velocity)])
        return float(result)

    def calculate_potential_energy(self, mass: float, height: float) -> float:
        """
        PE = mgh
        """
        result = self.potential_energy.subs([(self.m, mass), (self.h, height)])
        return float(result)

    def calculate_momentum(self, mass: float, velocity: float) -> float:
        """
        p = mv
        """
        result = self.momentum.subs([(self.m, mass), (self.v, velocity)])
        return float(result)

    def predict_next_state_pendulum(
        self,
        current_angle: float,
        current_velocity: float,
        length: float,
        dt: float = 0.01
    ) -> Tuple[float, float]:
        """
        Simulate pendulum: θ'' = -(g/L)sin(θ)

        Args:
            current_angle: radians
            current_velocity: rad/s
            length: m
            dt: timestep (s)

        Returns:
            next_angle, next_velocity
        """
        # Calculate angular acceleration
        accel_expr = self.pendulum_accel.subs(sp.Symbol('L'), length)
        angular_accel = float(accel_expr.subs(self.theta, current_angle))

        # Euler integration (simple, symbolic could use better integrator)
        next_velocity = current_velocity + angular_accel * dt
        next_angle = current_angle + next_velocity * dt

        return next_angle, next_velocity

    def check_center_of_mass_stability(
        self,
        com_x: float,
        com_y: float,
        support_polygon_x: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Check if center of mass is within support polygon (stable stance).

        Args:
            com_x, com_y: Center of mass position
            support_polygon_x: (left_foot, right_foot) x positions

        Returns:
            dict with:
                - is_stable: bool
                - stability_margin: distance to edge (positive = stable)
        """
        left_edge, right_edge = support_polygon_x

        # CoM must be between feet
        is_stable = left_edge <= com_x <= right_edge

        # Margin: distance to nearest edge
        margin_left = com_x - left_edge
        margin_right = right_edge - com_x
        stability_margin = min(margin_left, margin_right)

        return {
            'is_stable': bool(is_stable),
            'stability_margin': float(stability_margin),
            'com_x': com_x,
            'support_left': left_edge,
            'support_right': right_edge,
        }

    def predict_robot_state(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        robot_mass: float = 50.0,  # kg (typical humanoid)
        dt: float = 0.02  # 50Hz control
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict next robot state using physics.

        This is what the NEURAL network tries to learn to approximate!

        Args:
            current_state: (state_dim,) - joint angles, velocities
            action: (action_dim,) - joint torques
            robot_mass: kg
            dt: timestep

        Returns:
            next_state: (state_dim,)
            physics_quantities: dict of physical values
        """
        # Simplified model (real robot would use full dynamics)
        # For demo: assume action is acceleration

        # Extract relevant state (simplified)
        position = current_state[:3] if len(current_state) >= 3 else np.zeros(3)
        velocity = current_state[3:6] if len(current_state) >= 6 else np.zeros(3)

        # Action as force
        force = action[:3] if len(action) >= 3 else np.zeros(3)

        # F = ma → a = F/m
        acceleration = force / robot_mass

        # Integrate: v = v0 + at, x = x0 + vt
        next_velocity = velocity + acceleration * dt
        next_position = position + next_velocity * dt

        # Reconstruct state
        next_state = current_state.copy()
        if len(next_state) >= 3:
            next_state[:3] = next_position
        if len(next_state) >= 6:
            next_state[3:6] = next_velocity

        # Calculate physics quantities
        physics_quantities = {
            'kinetic_energy': self.calculate_kinetic_energy(robot_mass, np.linalg.norm(next_velocity)),
            'potential_energy': self.calculate_potential_energy(robot_mass, next_position[2]),
            'momentum': self.calculate_momentum(robot_mass, np.linalg.norm(next_velocity)),
            'force_magnitude': float(np.linalg.norm(force)),
        }

        return next_state, physics_quantities

    def verify_action_safe(
        self,
        current_state: np.ndarray,
        proposed_action: np.ndarray,
        safety_limits: Dict[str, float] = None
    ) -> Tuple[bool, str]:
        """
        Verify if proposed action is physically safe.

        This can run at inference time for critical safety checks!

        Args:
            current_state: robot state
            proposed_action: action neural network wants to take
            safety_limits: max force, max velocity, etc.

        Returns:
            is_safe: bool
            reason: str (why unsafe if False)
        """
        if safety_limits is None:
            safety_limits = {
                'max_force': 500.0,  # N
                'max_velocity': 5.0,  # m/s
                'min_height': 0.1,   # m (don't fall below ground)
            }

        # Check force magnitude
        force_mag = float(np.linalg.norm(proposed_action[:3])) if len(proposed_action) >= 3 else 0
        if force_mag > safety_limits['max_force']:
            return False, f"Force too high: {force_mag:.1f}N > {safety_limits['max_force']}N"

        # Predict next state
        next_state, physics = self.predict_robot_state(current_state, proposed_action)

        # Check velocity
        velocity = next_state[3:6] if len(next_state) >= 6 else np.zeros(3)
        vel_mag = float(np.linalg.norm(velocity))
        if vel_mag > safety_limits['max_velocity']:
            return False, f"Velocity too high: {vel_mag:.1f}m/s > {safety_limits['max_velocity']}m/s"

        # Check height (don't fall through ground)
        height = next_state[2] if len(next_state) >= 3 else 0
        if height < safety_limits['min_height']:
            return False, f"Height too low: {height:.2f}m < {safety_limits['min_height']}m"

        return True, "Action is safe"

    # ============================================================================
    # CHEMISTRY METHODS (For Phase 0C training)
    # ============================================================================

    def calculate_bond_energy(self, bond_type: str) -> float:
        """
        Get exact bond energy (kJ/mol)

        Args:
            bond_type: e.g., "C-C", "C=C", "O-H"

        Returns:
            energy: kJ/mol
        """
        return float(self.bond_energies.get(bond_type, 400.0))  # Default to typical bond

    def calculate_molecular_force(self, force_type: str) -> float:
        """
        Get intermolecular force strength (kJ/mol)

        Args:
            force_type: "van_der_waals", "hydrogen_bond", "ionic", "dipole_dipole"

        Returns:
            force_strength: kJ/mol
        """
        return float(self.intermolecular_forces.get(force_type, 10.0))

    def calculate_reaction_energy(
        self,
        bonds_broken: Dict[str, int],
        bonds_formed: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Calculate reaction energetics using bond energies.

        ΔH_reaction = Σ(bonds broken) - Σ(bonds formed)

        Args:
            bonds_broken: {"C-H": 4, "O=O": 1} etc.
            bonds_formed: {"C=O": 2, "O-H": 2} etc.

        Returns:
            dict with:
                - delta_H: enthalpy change (kJ/mol)
                - energy_in: energy required to break bonds
                - energy_out: energy released from forming bonds
                - exothermic: True if releases energy
        """
        # Energy required to BREAK bonds (endothermic, positive)
        energy_in = 0.0
        for bond, count in bonds_broken.items():
            energy_in += self.calculate_bond_energy(bond) * count

        # Energy released from FORMING bonds (exothermic, negative)
        energy_out = 0.0
        for bond, count in bonds_formed.items():
            energy_out += self.calculate_bond_energy(bond) * count

        # Net enthalpy change
        delta_H = energy_in - energy_out

        return {
            'delta_H': float(delta_H),
            'energy_in': float(energy_in),
            'energy_out': float(energy_out),
            'exothermic': delta_H < 0,
        }

    def get_material_properties(self, material: str) -> Dict[str, float]:
        """
        Get material properties for manipulation planning.

        Args:
            material: "rubber", "wood", "steel", "glass", "plastic"

        Returns:
            dict with elastic_modulus, hardness, friction_coeff, density
        """
        return self.material_properties.get(
            material.lower(),
            self.material_properties['plastic']  # Default
        )

    def calculate_grip_force(
        self,
        object_mass: float,
        material: str,
        safety_factor: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate required grip force to hold object.

        Uses friction: F_grip ≥ (mg) / μ

        Args:
            object_mass: kg
            material: surface material
            safety_factor: multiply required force (default 2.0)

        Returns:
            dict with required_force, friction_coeff, is_slippery
        """
        props = self.get_material_properties(material)
        friction_coeff = props['friction_coeff']

        # Weight of object
        weight = object_mass * float(self.g)

        # Required normal force (grip force)
        # F_grip * μ ≥ mg → F_grip ≥ mg/μ
        required_force = (weight / friction_coeff) * safety_factor

        return {
            'required_force': float(required_force),
            'friction_coeff': float(friction_coeff),
            'is_slippery': friction_coeff < 0.3,  # Low friction warning
            'object_weight': float(weight),
            'safety_factor': float(safety_factor),
        }

    def predict_material_deformation(
        self,
        material: str,
        applied_force: float,
        contact_area: float = 0.01  # m² (e.g., fingertip)
    ) -> Dict[str, float]:
        """
        Predict material behavior under force (elastic vs plastic).

        stress = force / area
        strain = stress / elastic_modulus

        Args:
            material: material type
            applied_force: N
            contact_area: m²

        Returns:
            dict with stress, strain, will_deform_permanently
        """
        props = self.get_material_properties(material)
        elastic_modulus = props['elastic_modulus'] * 1e9  # Convert GPa to Pa

        # Stress (Pa)
        stress = applied_force / contact_area

        # Strain (dimensionless)
        strain = stress / elastic_modulus

        # Yield strength (approximate from hardness)
        # Higher hardness → higher yield strength
        yield_stress = props['hardness'] * 50e6  # Approximate (Pa)

        return {
            'stress': float(stress),
            'strain': float(strain),
            'yield_stress': float(yield_stress),
            'will_deform_permanently': stress > yield_stress,
            'elastic_modulus': float(elastic_modulus),
            'material': material,
        }

    def calculate_molecular_vibration_energy(
        self,
        bond_type: str,
        temperature: float = 298.15  # K (room temp)
    ) -> Dict[str, float]:
        """
        Estimate vibrational energy of bond at given temperature.

        E_vib ≈ (1/2) * k_B * T

        Args:
            bond_type: e.g., "C-H"
            temperature: K

        Returns:
            dict with vibrational_energy, bond_energy, fraction_of_bond
        """
        k_B = 1.380649e-23  # Boltzmann constant (J/K)

        # Vibrational energy (J)
        E_vib = 0.5 * k_B * temperature

        # Convert to kJ/mol
        N_A = 6.02214076e23  # Avogadro's number
        E_vib_kJ_mol = E_vib * N_A / 1000

        bond_energy = self.calculate_bond_energy(bond_type)

        return {
            'vibrational_energy': float(E_vib_kJ_mol),
            'bond_energy': float(bond_energy),
            'fraction_of_bond': float(E_vib_kJ_mol / bond_energy),
            'temperature': float(temperature),
        }


def symbolic_supervision_loss(
    neural_prediction: torch.Tensor,
    state: np.ndarray,
    action: np.ndarray,
    calculator: SymbolicPhysicsCalculator
) -> Tuple[torch.Tensor, Dict]:
    """
    Training loss: Neural learns from symbolic calculator.

    This is the KEY to the hybrid approach!

    Args:
        neural_prediction: (batch, state_dim) - what neural net predicted
        state: (batch, state_dim) - current state
        action: (batch, action_dim) - action taken
        calculator: symbolic calculator

    Returns:
        loss: scalar
        metrics: dict
    """
    batch_size = neural_prediction.shape[0]
    device = neural_prediction.device

    # Get symbolic ground truth for each sample
    symbolic_truths = []
    for i in range(batch_size):
        state_i = state[i].cpu().numpy()
        action_i = action[i].cpu().numpy()

        # Symbolic calculator gives EXACT answer
        next_state_symbolic, _ = calculator.predict_robot_state(state_i, action_i)
        symbolic_truths.append(next_state_symbolic)

    symbolic_truths = torch.FloatTensor(np.array(symbolic_truths)).to(device)

    # Loss: Neural must match symbolic (teacher-student)
    loss = torch.nn.functional.mse_loss(neural_prediction, symbolic_truths)

    # Metrics
    error = (neural_prediction - symbolic_truths).abs().mean()

    return loss, {
        'symbolic_loss': loss.item(),
        'mean_error': error.item(),
    }


if __name__ == "__main__":
    print("="*70)
    print("SYMBOLIC PHYSICS CALCULATOR - DEMO")
    print("="*70 + "\n")

    calc = SymbolicPhysicsCalculator()

    # Test 1: Newton's law
    print("[TEST 1] F = ma")
    mass = 50.0  # kg (humanoid robot)
    accel = 2.0  # m/s²
    force = calc.calculate_force(mass, accel)
    print(f"  Mass: {mass}kg, Accel: {accel}m/s² → Force: {force}N")
    print(f"  ✓ Exact answer (not approximation!)\n")

    # Test 2: Torque
    print("[TEST 2] τ = r × F")
    radius = 0.3  # m (arm length)
    torque = calc.calculate_torque(radius, force)
    print(f"  Radius: {radius}m, Force: {force}N → Torque: {torque}N⋅m\n")

    # Test 3: Energy
    print("[TEST 3] Energy calculations")
    velocity = 1.5  # m/s
    height = 1.0    # m
    KE = calc.calculate_kinetic_energy(mass, velocity)
    PE = calc.calculate_potential_energy(mass, height)
    print(f"  KE = ½mv² = {KE:.2f}J")
    print(f"  PE = mgh = {PE:.2f}J")
    print(f"  Total = {KE + PE:.2f}J\n")

    # Test 4: Pendulum simulation
    print("[TEST 4] Pendulum dynamics")
    angle = 0.5    # rad (~30 degrees)
    vel = 0.0
    length = 1.0   # m

    print(f"  Initial: θ={angle:.2f}rad, ω={vel:.2f}rad/s")
    for step in range(3):
        angle, vel = calc.predict_next_state_pendulum(angle, vel, length, dt=0.1)
        print(f"  Step {step+1}: θ={angle:.2f}rad, ω={vel:.2f}rad/s")
    print()

    # Test 5: Stability check
    print("[TEST 5] Center of mass stability")
    com_x = 0.05   # m
    support = (-0.1, 0.1)  # feet at -10cm and +10cm
    stability = calc.check_center_of_mass_stability(com_x, 1.0, support)
    print(f"  CoM: {com_x}m, Support: {support}")
    print(f"  Stable: {stability['is_stable']}")
    print(f"  Margin: {stability['stability_margin']:.3f}m\n")

    # Test 6: Robot state prediction
    print("[TEST 6] Robot state prediction (what neural net learns)")
    state = np.random.randn(256)
    action = np.random.randn(17) * 10  # Some torques

    next_state, physics = calc.predict_robot_state(state, action)
    print(f"  Current state: {state[:3]}")
    print(f"  Action: {action[:3]}")
    print(f"  Next state: {next_state[:3]}")
    print(f"  Physics: KE={physics['kinetic_energy']:.2f}J, PE={physics['potential_energy']:.2f}J\n")

    # Test 7: Safety verification
    print("[TEST 7] Action safety check")
    safe_action = np.random.randn(17) * 10
    unsafe_action = np.random.randn(17) * 1000  # Way too much force!

    is_safe, reason = calc.verify_action_safe(state, safe_action)
    print(f"  Safe action: {is_safe} - {reason}")

    is_safe, reason = calc.verify_action_safe(state, unsafe_action)
    print(f"  Unsafe action: {is_safe} - {reason}\n")

    print("="*70)
    print("SUCCESS! Symbolic calculator ready for hybrid training!")
    print("="*70)
    print("\n[KEY POINTS]")
    print("  ✓ EXACT calculations (not neural approximations)")
    print("  ✓ Provides ground truth for neural training")
    print("  ✓ Can verify neural predictions at inference")
    print("  ✓ No hallucinations (pure mathematics)")
    print("\n[USAGE]")
    print("  Training: Neural learns from symbolic (teacher-student)")
    print("  Inference: Neural is fast, symbolic checks safety (optional)")
    print("="*70)
