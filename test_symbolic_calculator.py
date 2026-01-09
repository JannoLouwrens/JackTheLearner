import numpy as np
from SymbolicCalculator import SymbolicPhysicsCalculator
import torch

def run_test():
    """
    Tests the SymbolicPhysicsCalculator with random inputs and displays
    the data in both neural-net-friendly and human-friendly formats.
    """
    print("="*70)
    print("INITIALIZING SYMBOLIC CALCULATOR TEST (RANDOMIZED)")
    print("="*70 + "\n")

    # 1. Initialize the calculator
    calc = SymbolicPhysicsCalculator()

    # 2. Generate random inputs, just like in TRAIN_PHYSICS.py
    print("[STEP 1] Generating random state and action...\n")
    state_dim = 256  # As used in the trainer
    action_dim = 17  # As used in the trainer
    
    # Generate random state (joint angles, velocities) and scale
    state = np.random.randn(state_dim).astype(np.float32) * 0.5
    
    # Generate random action (torques/forces) and scale
    action = np.random.randn(action_dim).astype(np.float32) * 50.0
    
    # Expand state to 348 for SymbolicCalculator compatibility
    full_state = np.zeros(348, dtype=np.float32)
    full_state[:state_dim] = state
    
    # 3. Use the Symbolic Calculator to get the "ground truth" answer
    print("[STEP 2] Calculating ground truth with Symbolic Calculator...\n")
    
    # Get the next state and core physics quantities
    next_state_full, physics_dict = calc.predict_robot_state(full_state, action)
    next_state = next_state_full[:state_dim].astype(np.float32)

    # Also calculate chemistry/material properties for the full physics vector
    object_mass = np.random.uniform(0.1, 2.0)
    materials = ["rubber", "wood", "steel", "glass", "plastic"]
    material = np.random.choice(materials)
    grip_data = calc.calculate_grip_force(object_mass, material)
    material_props = calc.get_material_properties(material)

    # Construct the final 10-dimensional physics vector
    physics_vector = np.array([
        physics_dict['kinetic_energy'],
        physics_dict['potential_energy'],
        physics_dict['kinetic_energy'] + physics_dict['potential_energy'],
        physics_dict['momentum'],
        physics_dict['force_magnitude'],
        0.0,  # Torque placeholder
        0.0,  # Angular momentum placeholder
        0.0,  # Stability placeholder
        float(grip_data['friction_coeff']),
        float(material_props['elastic_modulus']),
    ], dtype=np.float32)

    # Convert to PyTorch tensors to perfectly mimic the training data format
    state_t = torch.from_numpy(state)
    action_t = torch.from_numpy(action)
    next_state_t = torch.from_numpy(next_state)
    physics_t = torch.from_numpy(physics_vector)

    # 4. Display the results in the two requested formats
    
    # --- NEURAL NET FORMAT ---
    print("="*70)
    print("FORMAT 1: NEURAL NETWORK VIEW")
    print("This is the raw numerical data the model sees for one sample.")
    print("="*70)
    print("\n--- THE 'QUESTION' (INPUT) ---")
    print(f"state:    Tensor of shape {state_t.shape}, dtype {state_t.dtype}")
    print(f"          Data: {state_t[:4]}...")
    print(f"\naction:   Tensor of shape {action_t.shape}, dtype {action_t.dtype}")
    print(f"          Data: {action_t[:4]}...")
    
    print("\n--- THE 'ANSWER' (GROUND TRUTH OUTPUT) ---")
    print(f"next_state: Tensor of shape {next_state_t.shape}, dtype {next_state_t.dtype}")
    print(f"            Data: {next_state_t[:4]}...")
    print(f"\nphysics:    Tensor of shape {physics_t.shape}, dtype {physics_t.dtype}")
    print(f"            Data: {physics_t}")
    print("\n")


    # --- FRIENDLY FORMAT ---
    print("="*70)
    print("FORMAT 2: HUMAN-FRIENDLY VIEW")
    print("This is the same data, but explained.")
    print("="*70)
    print("\n--- THE 'SCENARIO' (INPUT) ---")
    print("Current State:")
    print(f"  - Position (xyz): {state[:3]}")
    print(f"  - Velocity (xyz): {state[3:6]}")
    print("Proposed Action:")
    print(f"  - Applied Force (xyz): {action[:3]}")

    print("\n--- THE 'PREDICTION' (GROUND TRUTH OUTPUT) ---")
    print("Predicted Next State:")
    print(f"  - Next Position (xyz): {next_state[:3]}")
    print(f"  - Next Velocity (xyz): {next_state[3:6]}")
    print("\nPredicted Physics & Material Properties:")
    print(f"  - Kinetic Energy:      {physics_vector[0]:.4f} J")
    print(f"  - Potential Energy:    {physics_vector[1]:.4f} J")
    print(f"  - Total Energy:        {physics_vector[2]:.4f} J")
    print(f"  - Momentum:            {physics_vector[3]:.4f}")
    print(f"  - Force Magnitude:     {physics_vector[4]:.4f} N")
    print(f"  - Friction Coeff (from '{material}'): {physics_vector[8]:.4f} (unitless)")
    print(f"  - Elastic Modulus (from '{material}'): {physics_vector[9]:.4f} GPa")
    print("\n")

    print("="*70)
    print("RANDOMIZED TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_test()
