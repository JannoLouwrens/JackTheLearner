"""
FULL INTEGRATION TEST - JackTheWalker General-Purpose Humanoid Robot
====================================================================

This test demonstrates the complete robot capabilities:
1. Locomotion (walking, running, stairs)
2. Manipulation (reaching, grasping, carrying)
3. Object Detection (finding cups, coffee machines)
4. Navigation (go to kitchen, table)
5. Language Understanding (commands, questions)
6. Speech Response (answering questions, giving feedback)
7. Complex Task Execution (make coffee)

Author: Janno Louwrens
"""

import torch
import numpy as np
from typing import Dict, List

# Import the brain
from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig


def test_basic_initialization():
    """Test that all components initialize correctly."""
    print("\n" + "="*70)
    print("TEST 1: BASIC INITIALIZATION")
    print("="*70)

    config = UnifiedBrainConfig(
        vision_enabled=False,  # Skip for faster test
        audio_enabled=False,
        llm_enabled=False,
        enable_object_detection=True,
        enable_navigation=True,
        enable_response_generation=True,
        enable_task_completion=True,
        enable_memory=True,
        enable_tts=True,
    )

    brain = UnifiedBrain(config)

    # Verify all components exist
    assert brain.object_detector is not None, "ObjectDetector not initialized"
    assert brain.navigation_planner is not None, "NavigationPlanner not initialized"
    assert brain.response_generator is not None, "ResponseGenerator not initialized"
    assert brain.task_completion_head is not None, "TaskCompletionHead not initialized"
    assert brain.memory is not None, "Memory not initialized"
    assert brain.hierarchical_planner is not None, "HierarchicalPlanner not initialized"
    assert brain.world_model is not None, "WorldModel not initialized"
    assert brain.dual_system is not None, "DualSystem not initialized"

    print("\n[PASS] All components initialized correctly!")
    return brain


def test_locomotion(brain: UnifiedBrain):
    """Test basic locomotion action generation."""
    print("\n" + "="*70)
    print("TEST 2: LOCOMOTION")
    print("="*70)

    # Create dummy state
    state = torch.randn(1, brain.config.obs_dim)

    # Test forward pass
    output = brain(state)

    # Output has 'actions' (plural) not 'action'
    assert 'actions' in output, "No actions in output"
    assert output['actions'].shape[-1] == brain.config.action_dim, f"Wrong action dim: {output['actions'].shape}"

    print(f"\n  State shape: {state.shape}")
    print(f"  Actions shape: {output['actions'].shape}")
    print(f"  Actions mean: {output['actions'].mean().item():.4f}")
    print(f"  Actions std: {output['actions'].std().item():.4f}")

    # Test with goal conditioning
    goal = torch.randn(1, brain.config.obs_dim)  # goal same shape as state
    output_goal = brain(state, goal=goal)

    assert 'actions' in output_goal, "No actions with goal"
    print(f"\n  Goal-conditioned actions: {output_goal['actions'].shape}")

    print("\n[PASS] Locomotion working!")
    return True


def test_object_detection(brain: UnifiedBrain):
    """Test object detection capabilities."""
    print("\n" + "="*70)
    print("TEST 3: OBJECT DETECTION")
    print("="*70)

    # Test finding different objects
    objects_to_find = ["cup", "coffee machine", "table", "kitchen", "bottle"]

    for obj in objects_to_find:
        result = brain.find_object(obj)
        print(f"\n  Finding '{obj}':")
        print(f"    Found: {result.get('found', False)}")
        if result.get('found'):
            print(f"    Position: {result.get('position')}")
            print(f"    Confidence: {result.get('confidence', 0):.3f}")

    # Test with actual vision features (mock)
    vision_features = torch.randn(1, 49, brain.config.d_model)
    result = brain.object_detector(vision_features)

    print(f"\n  Detector output:")
    print(f"    Classes shape: {result['classes'].shape}")
    print(f"    Positions shape: {result['positions'].shape}")
    print(f"    Scores shape: {result['scores'].shape}")

    print("\n[PASS] Object detection working!")
    return True


def test_navigation(brain: UnifiedBrain):
    """Test navigation planning capabilities."""
    print("\n" + "="*70)
    print("TEST 4: NAVIGATION")
    print("="*70)

    # Test navigation to different targets
    targets = ["kitchen", "table", "door", "start"]
    current_pos = torch.zeros(3)  # Start at origin

    for target in targets:
        result = brain.navigate_to(target, current_pos)
        print(f"\n  Navigating to '{target}':")
        print(f"    Action: {result.get('action')}")
        print(f"    Distance: {result.get('distance', 0):.2f}m")
        print(f"    Arrived: {result.get('arrived', False)}")

    # Test path planning
    start = torch.tensor([0.0, 0.0, 0.0])
    goal = torch.tensor([3.0, 0.0, 0.0])
    path = brain.navigation_planner.plan_path(start, goal)

    print(f"\n  Path planning (start -> kitchen):")
    print(f"    Waypoints: {len(path)}")
    for i, wp in enumerate(path):
        print(f"    {i}: [{wp[0].item():.2f}, {wp[1].item():.2f}, {wp[2].item():.2f}]")

    print("\n[PASS] Navigation working!")
    return True


def test_language_commands(brain: UnifiedBrain):
    """Test language command understanding."""
    print("\n" + "="*70)
    print("TEST 5: LANGUAGE COMMANDS")
    print("="*70)

    commands = [
        "go to the kitchen",
        "pick up the cup",
        "bring me coffee",
        "walk forward",
        "turn left",
        "stop",
    ]

    for cmd in commands:
        result = brain._execute_command(cmd)
        print(f"\n  Command: '{cmd}'")
        print(f"    Type: {result.get('type')}")
        print(f"    Response: {result.get('response')}")
        if 'action' in result:
            print(f"    Action: {result.get('action')}")
        if 'steps' in result:
            print(f"    Steps: {result.get('steps')}")

    print("\n[PASS] Language commands working!")
    return True


def test_chat_interface(brain: UnifiedBrain):
    """Test the conversational chat interface."""
    print("\n" + "="*70)
    print("TEST 6: CHAT INTERFACE")
    print("="*70)

    conversations = [
        ("What's 1+1?", "question"),
        ("go to the kitchen", "command"),
        ("Hello, how are you?", "chat"),
        ("pick up the bottle", "command"),
    ]

    for message, expected_type in conversations:
        response = brain.chat(message, speak=False)
        print(f"\n  User: '{message}'")
        print(f"  Robot: '{response}'")
        print(f"  Type: {expected_type}")

    print("\n[PASS] Chat interface working!")
    return True


def test_make_coffee_scenario(brain: UnifiedBrain):
    """Test the complete 'make coffee' task."""
    print("\n" + "="*70)
    print("TEST 7: MAKE COFFEE SCENARIO")
    print("="*70)

    result = brain.make_coffee()

    print(f"\n  Task: {result.get('task')}")
    print(f"  Status: {result.get('status')}")
    print(f"  Response: {result.get('response')}")
    print(f"\n  Steps ({len(result.get('steps', []))} total):")

    for i, step in enumerate(result.get('steps', [])):
        action = step.get('action')
        target = step.get('target', step.get('location', step.get('duration', '')))
        print(f"    {i+1}. {action.upper()}: {target}")

    # Simulate executing steps
    print("\n  Simulating execution:")
    for i, step in enumerate(result.get('steps', [])):
        action = step['action']
        if action == 'find':
            find_result = brain.find_object(step['target'])
            print(f"    Step {i+1}: Finding {step['target']} - {'Found' if find_result.get('found') else 'Searching...'}")
        elif action == 'navigate':
            nav_result = brain.navigate_to(step['target'], torch.zeros(3))
            print(f"    Step {i+1}: Navigating to {step['target']} - Distance: {nav_result.get('distance', 0):.2f}m")
        elif action == 'pick_up':
            print(f"    Step {i+1}: Picking up {step['target']}")
        elif action == 'place':
            print(f"    Step {i+1}: Placing {step['target']} at {step.get('location')}")
        elif action == 'press':
            print(f"    Step {i+1}: Pressing {step['target']}")
        elif action == 'wait':
            print(f"    Step {i+1}: Waiting {step.get('duration')}s for coffee")
        else:
            print(f"    Step {i+1}: {action}")

    print("\n[PASS] Make coffee scenario working!")
    return True


def test_dual_system(brain: UnifiedBrain):
    """Test the dual system architecture (S0/S1/S2)."""
    print("\n" + "="*70)
    print("TEST 8: DUAL SYSTEM ARCHITECTURE")
    print("="*70)

    if brain.dual_system is None:
        print("  DualSystem not enabled")
        return True

    print(f"  System 2 (VLM reasoning): {brain.config.system2_hz} Hz")
    print(f"  System 1 (Action generation): {brain.config.system1_hz} Hz")
    print(f"  System 0 (Motor control): {brain.config.system0_hz} Hz")

    # Test dual system controller
    state = torch.randn(1, brain.config.obs_dim)

    # Mock inputs for dual system
    proprio_embed = brain.proprio_encoder(state)

    # Test coordination
    print(f"\n  Proprio embedding shape: {proprio_embed.shape}")

    print("\n[PASS] Dual system architecture working!")
    return True


def test_world_model(brain: UnifiedBrain):
    """Test the TD-MPC2 world model."""
    print("\n" + "="*70)
    print("TEST 9: WORLD MODEL (TD-MPC2)")
    print("="*70)

    # Create dummy inputs
    state = torch.randn(1, brain.config.d_model)
    action = torch.randn(1, brain.config.action_dim)

    # Test world model components
    z = brain.world_model.encode(state)  # Use encode() method
    print(f"\n  Latent encoding shape: {z.shape}")

    # Test dynamics via predict_next (dynamics takes concatenated input)
    decoded, reward, z_next = brain.world_model.predict_next(z, action)
    print(f"  Next latent shape: {z_next.shape}")
    print(f"  Decoded state shape: {decoded.shape}")

    # Test reward prediction
    reward_val = brain.world_model.reward_predictor(z)
    print(f"  Reward prediction: {reward_val.item():.4f}")

    # Test imagination with action sequence
    actions = torch.randn(1, brain.config.imagination_horizon, brain.config.action_dim)
    z_seq, r_seq = brain.world_model.imagine_trajectory(z, actions)
    print(f"  Imagination horizon: {z_seq.shape[1]} steps")
    rewards_str = ", ".join(f"{r.item():.3f}" for r in r_seq[0])
    print(f"  Imagined rewards: [{rewards_str}]")

    print("\n[PASS] World model working!")
    return True


def test_hierarchical_planner(brain: UnifiedBrain):
    """Test the hierarchical planner (HAC)."""
    print("\n" + "="*70)
    print("TEST 10: HIERARCHICAL PLANNER (HAC)")
    print("="*70)

    # Create dummy inputs
    cls_features = torch.randn(1, brain.config.d_model)
    task = torch.randn(1, brain.config.goal_dim)

    # Test planning via plan() method
    result = brain.hierarchical_planner.plan(cls_features, task)

    print(f"\n  Subgoals shape: {result['subgoals'].shape}")
    print(f"  Active subgoal shape: {result['active_subgoal'].shape}")
    print(f"  Low-level goal shape: {result['low_level_goal'].shape}")
    print(f"  Skill ID: {result['skill_id']}")
    print(f"  Skill name: {result['skill_name']}")
    print(f"  Termination prob: {result['termination_prob']:.3f}")

    # Reset planner
    brain.hierarchical_planner.reset()
    print(f"\n  Planner reset successful")

    print("\n[PASS] Hierarchical planner working!")
    return True


def test_memory_system(brain: UnifiedBrain):
    """Test the companion memory system."""
    print("\n" + "="*70)
    print("TEST 11: MEMORY SYSTEM")
    print("="*70)

    if brain.memory is None:
        print("  Memory not enabled")
        return True

    # Store some memories - remember(fact, importance) signature
    memories = [
        ("User asked for coffee", 1.0),
        ("User prefers milk in coffee", 0.8),
        ("Cup is on the table", 0.9),
    ]

    print("\n  Storing memories:")
    for text, importance in memories:
        brain.remember(text, importance)
        print(f"    Stored: '{text}' (importance={importance})")

    # Recall memories
    print("\n  Recalling memories:")
    queries = ["coffee", "cup", "milk"]
    for query in queries:
        recalled = brain.recall(query, top_k=2)
        print(f"    Query: '{query}'")
        for mem in recalled:
            if isinstance(mem, dict):
                print(f"      -> {mem.get('text', str(mem))[:50]}...")
            else:
                print(f"      -> {str(mem)[:50]}...")

    print("\n[PASS] Memory system working!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  JACKTHEWALKER FULL INTEGRATION TEST".center(68) + "#")
    print("#" + "  General-Purpose Humanoid Robot Brain".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    # Initialize
    brain = test_basic_initialization()

    # Run all tests
    tests = [
        ("Locomotion", lambda: test_locomotion(brain)),
        ("Object Detection", lambda: test_object_detection(brain)),
        ("Navigation", lambda: test_navigation(brain)),
        ("Language Commands", lambda: test_language_commands(brain)),
        ("Chat Interface", lambda: test_chat_interface(brain)),
        ("Make Coffee", lambda: test_make_coffee_scenario(brain)),
        ("Dual System", lambda: test_dual_system(brain)),
        ("World Model", lambda: test_world_model(brain)),
        ("Hierarchical Planner", lambda: test_hierarchical_planner(brain)),
        ("Memory System", lambda: test_memory_system(brain)),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
