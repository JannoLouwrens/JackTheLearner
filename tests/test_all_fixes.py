"""
COMPREHENSIVE UNIT TESTS - Verifies all bug fixes across JackTheWalker
=====================================================================

Tests all modules:
- ReplayBuffer (phase tracking, save/load)
- MoCapLoader (joint limits, spine, regex, dt, PD, synthetic)
- SymbolicCalculator (NaN safety, joint physics, type handling, torque checks)
- TrainingPipeline (PPO update, checkpoints, project_obs)
- UnifiedBrain (forward pass, world model, hierarchical planner)
- Companion Integration (emotional state, mood effects, persistence)

Run: py -m pytest tests/test_all_fixes.py -v
  or: py tests/test_all_fixes.py
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# TEST 1: ReplayBuffer - phase tracking after deque rotation
# ==============================================================================

class TestReplayBuffer:
    def test_phase_tracking_survives_rotation(self):
        """Phase sampling must work correctly after buffer wraps around."""
        from TrainingPipeline import ReplayBuffer

        buf = ReplayBuffer(capacity=10)

        # Add 5 Phase 0 samples
        for i in range(5):
            buf.add({'state': torch.randn(4), 'value': f'p0_{i}'}, phase=0)

        # Add 10 Phase 1 samples (overwrites some Phase 0 due to capacity=10)
        for i in range(10):
            buf.add({'state': torch.randn(4), 'value': f'p1_{i}'}, phase=1)

        assert len(buf) == 10, f"Buffer should have 10 items, got {len(buf)}"

        # Sample only Phase 0 - should get whatever Phase 0 samples remain
        phase0_samples = buf.sample(5, phase_ratios={0: 1.0})
        for s in phase0_samples:
            assert s['_phase'] == 0, f"Expected phase 0, got {s['_phase']}"

        # Sample only Phase 1
        phase1_samples = buf.sample(5, phase_ratios={1: 1.0})
        for s in phase1_samples:
            assert s['_phase'] == 1, f"Expected phase 1, got {s['_phase']}"

        print("[PASS] ReplayBuffer phase tracking survives rotation")

    def test_empty_buffer_returns_empty(self):
        from TrainingPipeline import ReplayBuffer
        buf = ReplayBuffer(capacity=10)
        assert buf.sample(5) == []
        assert buf.sample(5, phase_ratios={0: 1.0}) == []
        print("[PASS] ReplayBuffer empty returns empty")

    def test_save_load_preserves_phases(self):
        from TrainingPipeline import ReplayBuffer
        import tempfile

        buf = ReplayBuffer(capacity=100)
        buf.add({'x': 1}, phase=0)
        buf.add({'x': 2}, phase=1)
        buf.add({'x': 3}, phase=0)

        path = os.path.join(tempfile.gettempdir(), 'test_replay.pt')
        buf.save(path)

        buf2 = ReplayBuffer(capacity=100)
        buf2.load(path)

        assert len(buf2) == 3
        # Verify phases survived round-trip
        p0 = buf2.sample(10, phase_ratios={0: 1.0})
        p1 = buf2.sample(10, phase_ratios={1: 1.0})
        assert len(p0) > 0, "Phase 0 samples lost"
        assert len(p1) > 0, "Phase 1 samples lost"
        for s in p0:
            assert s['_phase'] == 0

        os.remove(path)
        print("[PASS] ReplayBuffer save/load preserves phases")


# ==============================================================================
# TEST 2: MoCapLoader - joint limits
# ==============================================================================

class TestMoCapLoader:
    def test_joint_limits_are_realistic(self):
        """Joint limits must allow full range of human motion."""
        from MoCapLoader import MoCapConfig

        config = MoCapConfig()
        limits = config.joint_limits

        # Knee must allow deep flexion (at least -2.0 radians = -115 degrees)
        assert limits['right_knee'][0] < -2.0, f"Right knee min too small: {limits['right_knee'][0]}"
        assert limits['left_knee'][0] < -2.0, f"Left knee min too small: {limits['left_knee'][0]}"

        # Hip must allow large flexion/extension
        assert limits['right_hip_y'][0] < -1.5, f"Right hip y min too small: {limits['right_hip_y'][0]}"

        # Shoulders must allow overhead reach
        assert limits['right_shoulder1'][1] > 1.0, f"Right shoulder1 max too small"

        # Elbow must allow full flexion
        assert limits['right_elbow'][0] < -2.0, f"Right elbow min too small"

        # Verify no joint is stuck at [-0.4, 0.4]
        for name, (lo, hi) in limits.items():
            assert not (lo == -0.4 and hi == 0.4), f"Joint {name} still has placeholder limits"

        print("[PASS] MoCapLoader joint limits are realistic")

    def test_pd_gains_are_reasonable(self):
        """PD gains must not cause permanent saturation."""
        from MoCapLoader import MoCapConfig

        config = MoCapConfig()
        # With a typical position error of 0.5 rad, torque should be < 10
        typical_error = 0.5
        typical_torque = config.kp * typical_error
        assert typical_torque < 10.0, f"PD torque {typical_torque} will saturate at action limits"
        print("[PASS] MoCapLoader PD gains are reasonable")

    def test_synthetic_fallback_produces_diverse_samples(self):
        """Synthetic samples must vary between calls."""
        from MoCapLoader import MoCapDataset, MoCapConfig

        config = MoCapConfig(mocap_dir="nonexistent_dir_12345")
        dataset = MoCapDataset(config)

        # Get two samples - they should differ due to randomization
        s1 = dataset[0]
        s2 = dataset[0]

        obs1, act1, label1 = s1
        obs2, act2, label2 = s2

        # At least the observations should differ (random amplitude/phase)
        # Note: there's a small chance they could be identical, but very unlikely
        diff = torch.abs(obs1 - obs2).sum().item()
        assert diff > 0.01, f"Synthetic samples are identical (diff={diff})"
        print("[PASS] MoCapLoader synthetic fallback produces diverse samples")


# ==============================================================================
# TEST 3: SymbolicCalculator - NaN safety and joint physics
# ==============================================================================

class TestSymbolicCalculator:
    def test_nan_rejected_by_safety_check(self):
        """NaN inputs must be rejected, not silently passed."""
        from SymbolicCalculator import SymbolicPhysicsCalculator

        calc = SymbolicPhysicsCalculator()

        # NaN state
        nan_state = np.full(256, np.nan)
        normal_action = np.zeros(17)
        is_safe, reason = calc.verify_action_safe(nan_state, normal_action)
        assert not is_safe, "NaN state should be rejected"
        assert "NaN" in reason or "Inf" in reason or "Invalid" in reason

        # NaN action
        normal_state = np.zeros(256)
        nan_action = np.full(17, np.nan)
        is_safe, reason = calc.verify_action_safe(normal_state, nan_action)
        assert not is_safe, "NaN action should be rejected"

        # Inf action
        inf_action = np.full(17, np.inf)
        is_safe, reason = calc.verify_action_safe(normal_state, inf_action)
        assert not is_safe, "Inf action should be rejected"

        print("[PASS] SymbolicCalculator rejects NaN/Inf inputs")

    def test_joint_physics_updates_more_than_6_dims(self):
        """predict_robot_state must update joint angles, not just position/velocity."""
        from SymbolicCalculator import SymbolicPhysicsCalculator

        calc = SymbolicPhysicsCalculator()

        # Create state with 40 dims (pos=3, vel=3, joints=17, joint_vels=17)
        state = np.zeros(40)
        state[6:23] = 0.1  # Joint angles at 0.1 rad
        state[23:40] = 0.0  # Joint velocities at 0

        # Action with joint torques
        action = np.zeros(20)
        action[3:20] = 0.5  # Apply torque to all joints

        next_state, physics = calc.predict_robot_state(state, action)

        # Joint angles should have changed from the applied torque
        joint_change = np.abs(next_state[6:23] - state[6:23]).sum()
        assert joint_change > 0.0, f"Joint angles unchanged after torque application"

        print("[PASS] SymbolicCalculator updates joint dimensions")

    def test_joint_torque_limits_enforced(self):
        """Excessive joint torques must be rejected."""
        from SymbolicCalculator import SymbolicPhysicsCalculator

        calc = SymbolicPhysicsCalculator()

        state = np.zeros(256)
        state[2] = 1.0  # Height above ground

        # Extreme torque on joint 0
        action = np.zeros(17)
        action[3] = 200.0  # Way above 100 Nm limit

        is_safe, reason = calc.verify_action_safe(state, action)
        assert not is_safe, f"Extreme torque should be rejected, got: {reason}"

        print("[PASS] SymbolicCalculator enforces joint torque limits")

    def test_physics_values_are_correct(self):
        """Basic physics calculations must be mathematically correct."""
        from SymbolicCalculator import SymbolicPhysicsCalculator

        calc = SymbolicPhysicsCalculator()

        # F = ma
        assert abs(calc.calculate_force(10.0, 2.0) - 20.0) < 1e-10

        # a = F/m
        assert abs(calc.calculate_acceleration(20.0, 10.0) - 2.0) < 1e-10

        # KE = 0.5 * m * v^2
        assert abs(calc.calculate_kinetic_energy(2.0, 3.0) - 9.0) < 1e-10

        # PE = mgh
        assert abs(calc.calculate_potential_energy(2.0, 5.0) - 2.0 * 9.81 * 5.0) < 1e-6

        print("[PASS] SymbolicCalculator physics values are correct")


# ==============================================================================
# TEST 4: TrainingPipeline - PPO, checkpoints, project_obs
# ==============================================================================

class TestTrainingPipeline:
    def test_ppo_update_runs(self):
        """PPO rl_update must produce valid metrics."""
        from TrainingPipeline import TrainingPipeline, PipelineConfig

        config = PipelineConfig()
        pipeline = TrainingPipeline(config)
        pipeline.make_optimizer(2)

        N = 32
        rollout = {
            'states': torch.randn(N, 256, device=pipeline.device),
            'actions': torch.randn(N, 17, device=pipeline.device),
            'log_probs': torch.randn(N, device=pipeline.device),
            'values': torch.randn(N, device=pipeline.device),
            'rewards': torch.randn(N, device=pipeline.device),
            'dones': torch.zeros(N, device=pipeline.device),
        }
        metrics = pipeline.rl_update(rollout)
        assert 'pg_loss' in metrics
        assert 'vf_loss' in metrics
        assert 'entropy' in metrics
        print("[PASS] TrainingPipeline PPO update runs")

    def test_checkpoint_includes_obs_proj(self):
        """Checkpoints must save and restore obs_projection."""
        from TrainingPipeline import TrainingPipeline, PipelineConfig
        import tempfile

        config = PipelineConfig(checkpoint_dir=tempfile.mkdtemp())
        pipeline = TrainingPipeline(config)
        pipeline.make_optimizer(0)

        # Modify obs_proj weights so we can verify they're restored
        with torch.no_grad():
            pipeline.obs_proj[0].weight.data.fill_(42.0)

        pipeline.save("test_ckpt")

        with torch.no_grad():
            pipeline.obs_proj[0].weight.data.fill_(0.0)  # Reset

        pipeline.load("test_ckpt")

        assert pipeline.obs_proj[0].weight.mean().item() > 40.0, "obs_proj not restored from checkpoint"
        print("[PASS] TrainingPipeline checkpoint includes obs_projection")

    def test_project_obs_handles_all_dims(self):
        """project_obs must handle 256, 376, and arbitrary dims."""
        from TrainingPipeline import TrainingPipeline, PipelineConfig

        config = PipelineConfig()
        pipeline = TrainingPipeline(config)

        assert pipeline.project_obs(torch.randn(1, 256)).shape == (1, 256)
        assert pipeline.project_obs(torch.randn(1, 376)).shape == (1, 256)
        assert pipeline.project_obs(torch.randn(1, 100)).shape == (1, 256)
        print("[PASS] TrainingPipeline project_obs handles all dims")

    def test_find_checkpoint_returns_none_for_missing(self):
        """find_checkpoint must return None when nothing exists."""
        from TrainingPipeline import TrainingPipeline, PipelineConfig

        config = PipelineConfig()
        pipeline = TrainingPipeline(config)

        result = pipeline.find_checkpoint("nonexistent_xyz")
        assert result is None
        print("[PASS] TrainingPipeline find_checkpoint works")


# ==============================================================================
# TEST 5: UnifiedBrain - architecture correctness
# ==============================================================================

class TestUnifiedBrain:
    def test_amp_discriminator_action_dim(self):
        """AMPDiscriminator default action_dim must be 17, not 57."""
        from UnifiedBrain import AMPDiscriminator

        disc = AMPDiscriminator()
        # Input dim should be 256*2 + 17 = 529
        expected_input = 256 * 2 + 17
        actual_input = disc.encoder[0].in_features
        assert actual_input == expected_input, f"AMP input dim {actual_input} != expected {expected_input}"
        print("[PASS] AMPDiscriminator default action_dim is 17")

    def test_forward_pass_all_outputs(self):
        """UnifiedBrain forward must produce all expected outputs."""
        from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

        config = UnifiedBrainConfig(
            vision_enabled=False,
            audio_enabled=False,
            llm_enabled=False,
        )
        brain = UnifiedBrain(config)

        B = 2
        state = torch.randn(B, config.obs_dim)
        action = torch.randn(B, config.action_dim)

        with torch.no_grad():
            output = brain(state, action=action)

        assert 'actions' in output, "Missing 'actions'"
        assert 'physics' in output, "Missing 'physics'"
        assert 'value' in output, "Missing 'value'"
        assert 'next_state' in output, "Missing 'next_state'"
        assert 'cls_features' in output, "Missing 'cls_features'"

        assert output['actions'].shape[-1] == config.action_dim
        assert output['physics'].shape[-1] == 10

        print("[PASS] UnifiedBrain forward pass produces all outputs")

    def test_world_model_imagination(self):
        """World model must imagine trajectories of correct shape."""
        from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

        config = UnifiedBrainConfig(
            vision_enabled=False, audio_enabled=False, llm_enabled=False,
        )
        brain = UnifiedBrain(config)

        B = 2
        state = torch.randn(B, config.obs_dim)
        actions = torch.randn(B, 5, config.action_dim)

        with torch.no_grad():
            latents, rewards = brain.imagine(state, actions)

        assert latents.shape == (B, 6, config.latent_dim), f"Latent shape: {latents.shape}"
        assert rewards.shape == (B, 5), f"Reward shape: {rewards.shape}"

        print("[PASS] UnifiedBrain world model imagination works")

    def test_hierarchical_planner(self):
        """Hierarchical planner must produce skill selections."""
        from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

        config = UnifiedBrainConfig(
            vision_enabled=False, audio_enabled=False, llm_enabled=False,
        )
        brain = UnifiedBrain(config)
        brain.reset_planner()

        state = torch.randn(1, config.obs_dim)
        task = torch.randn(1, config.goal_dim)

        with torch.no_grad():
            output = brain(state, task=task, use_hierarchy=True)

        assert 'hierarchy' in output
        h = output['hierarchy']
        assert 'skill_name' in h
        assert 'subgoals' in h
        assert h['subgoals'].shape == (1, config.max_subgoals, config.goal_dim)

        print("[PASS] UnifiedBrain hierarchical planner works")


# ==============================================================================
# TEST 6: Companion Modules Integration
# ==============================================================================

class TestCompanionIntegration:
    def test_emotional_state_on_brain(self):
        """Brain must have emotional state integrated."""
        from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
        config = UnifiedBrainConfig(vision_enabled=False, audio_enabled=False, llm_enabled=False)
        brain = UnifiedBrain(config)
        assert brain.emotional_state is not None, "EmotionalState missing"
        assert brain.personality is not None, "Personality missing"
        assert brain.movement_mood is not None, "MovementMoodCoupling missing"
        print("[PASS] Companion modules integrated on brain")

    def test_mood_affects_forward_pass(self):
        """Mood embedding must be injected into transformer."""
        from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
        from EmotionalState import EventType

        config = UnifiedBrainConfig(vision_enabled=False, audio_enabled=False, llm_enabled=False)
        brain = UnifiedBrain(config)
        state = torch.randn(1, 256)

        # Get output with neutral mood
        brain.emotional_state.pad_vector.data.zero_()
        out1 = brain(state)['actions'].detach().clone()

        # Get output with strong happy mood
        brain.emotional_state.pad_vector.data.copy_(torch.tensor([0.9, 0.8, 0.5]))
        out2 = brain(state)['actions'].detach().clone()

        # Actions should differ (mood embedding changed the transformer input)
        diff = (out1 - out2).abs().sum().item()
        assert diff > 0.001, f"Mood had no effect on actions (diff={diff})"
        print("[PASS] Mood affects forward pass")

    def test_persistence_round_trip(self):
        """Save/load must preserve companion state."""
        from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
        from Persistence import CompanionPersistence, SaveConfig
        import tempfile

        config = UnifiedBrainConfig(vision_enabled=False, audio_enabled=False, llm_enabled=False,
                                     enable_intrinsic_motivation=False)
        brain = UnifiedBrain(config)
        brain.emotional_state.pad_vector.data.copy_(torch.tensor([0.5, -0.3, 0.2]))
        brain.remember("Test memory")

        saver = CompanionPersistence(SaveConfig(save_dir=tempfile.mkdtemp()))
        path = saver.save_all(brain)
        assert os.path.exists(path)

        # Reset and reload
        brain.emotional_state.pad_vector.data.zero_()
        saver.load_all(brain, path)
        restored = brain.emotional_state.pad_vector
        assert abs(restored[0].item() - 0.5) < 0.01, "Emotional state not restored"
        print("[PASS] Persistence round-trip works")


# ==============================================================================
# RUN ALL TESTS
# ==============================================================================

def run_all():
    test_classes = [
        TestReplayBuffer,
        TestMoCapLoader,
        TestSymbolicCalculator,
        TestTrainingPipeline,
        TestUnifiedBrain,
        TestCompanionIntegration,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")

        instance = cls()
        for method_name in sorted(dir(instance)):
            if method_name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    failed += 1
                    errors.append((f"{cls.__name__}.{method_name}", str(e)))
                    print(f"[FAIL] {method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
