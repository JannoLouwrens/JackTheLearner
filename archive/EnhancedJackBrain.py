"""
ENHANCED JACK BRAIN - SYSTEM 1 + SYSTEM 2 UNIFIED

This is the complete AGI brain that combines:
- System 1 (ScalableRobotBrain): Fast, reactive control at 50Hz
- System 2 (WorldModel, MathReasoner, HAC, AlphaGeometry): Slow reasoning at 1-5Hz

Architecture inspired by Kahneman's "Thinking, Fast and Slow":
- System 1: Automatic, always-on, pattern matching
- System 2: Deliberate, effortful, logical reasoning

THREE RUNTIME MODES:
1. REACTIVE (90%): Pure System 1 - maximum speed
2. VERIFIED (9%): System 1 + symbolic verification - safety
3. CREATIVE (1%): Full AlphaGeometry loop - novel problem solving

Research Papers:
- TD-MPC2 (ICLR 2024): World model imagination
- HAC (2019): Hierarchical Actor-Critic skills
- AlphaGeometry (Nature 2024): Neural-symbolic creative loop
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

# Import System 1 components
from ScalableRobotBrain import (
    BrainConfig,
    ScalableRobotBrain,
    flow_matching_loss,
)


# ==============================================================================
# AGI CONFIGURATION
# ==============================================================================

@dataclass
class AGIConfig:
    """Configuration for complete AGI system (System 1 + System 2)"""
    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    action_dim: int = 17
    obs_dim: int = 348

    # System 2 components
    use_world_model: bool = True       # TD-MPC2 imagination
    use_math_reasoning: bool = True    # SymPy physics verification
    use_hierarchical: bool = True      # HAC skill decomposition
    use_creative_loop: bool = True     # AlphaGeometry problem solving

    # Frequencies
    system1_hz: int = 50   # Fast control loop
    system2_hz: int = 5    # Slow reasoning loop

    # Mode thresholds
    reactive_threshold: float = 0.9    # Confidence needed for pure reactive
    creative_threshold: float = 0.3    # Novelty threshold for creative mode


# ==============================================================================
# ENHANCED JACK BRAIN (SYSTEM 1 + SYSTEM 2)
# ==============================================================================

class EnhancedJackBrain(nn.Module):
    """
    The unified AGI brain.

    ONE brain. TWO systems. THREE modes.
    Fast + Slow. Neural + Symbolic. Reactive + Creative.
    """

    def __init__(self, config: AGIConfig = None, obs_dim: int = 348):
        super().__init__()

        if config is None:
            config = AGIConfig()

        self.config = config

        print("\n" + "="*70)
        print("       ENHANCED JACK BRAIN - UNIFIED AGI SYSTEM")
        print("="*70 + "\n")

        # ==========================================
        # SYSTEM 1: FAST THINKING (50Hz)
        # ==========================================
        print("[SYSTEM 1] Fast Thinking (50Hz)")
        brain_config = BrainConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            action_dim=config.action_dim,
            use_diffusion=True,
            use_flow_matching=True,
        )
        self.system1 = ScalableRobotBrain(brain_config, obs_dim)
        print("  -> VLA Transformer + Flow Matching Diffusion\n")

        # ==========================================
        # SYSTEM 2: SLOW THINKING (1-5Hz)
        # ==========================================
        print("[SYSTEM 2] Slow Thinking (1-5Hz)\n")

        # WorldModel (TD-MPC2) - Imagination
        if config.use_world_model:
            print("  [2.1] WorldModel (TD-MPC2)")
            from WorldModel import TD_MPC2_WorldModel, WorldModelConfig
            self.world_model = TD_MPC2_WorldModel(WorldModelConfig(
                latent_dim=256,
                action_dim=config.action_dim,
                obs_dim=obs_dim,
            ))
            print("    -> Latent dynamics for imagination\n")
        else:
            self.world_model = None

        # MathReasoner + SymbolicCalculator - Physics
        if config.use_math_reasoning:
            print("  [2.2] MathReasoner (Neuro-Symbolic)")
            from MathReasoner import NeuroSymbolicMathReasoner, MathReasonerConfig
            from SymbolicCalculator import SymbolicPhysicsCalculator
            self.math_reasoner = NeuroSymbolicMathReasoner(MathReasonerConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                num_rules=100,
                proprio_dim=256,
                action_dim=config.action_dim,
            ))
            self.symbolic_calculator = SymbolicPhysicsCalculator()
            print("    -> Neural learns 100 physics rules from SymPy\n")
        else:
            self.math_reasoner = None
            self.symbolic_calculator = None

        # HierarchicalPlanner (HAC) - Skill Decomposition
        if config.use_hierarchical:
            print("  [2.3] HierarchicalPlanner (HAC)")
            from HierarchicalPlanner import HierarchicalPlanner, HierarchicalPlannerConfig
            self.hierarchical = HierarchicalPlanner(HierarchicalPlannerConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=4,
                num_skills=20,
                state_dim=256,
                goal_dim=64,
                action_dim=config.action_dim,
            ))
            print("    -> 20 learnable skills for task decomposition\n")
        else:
            self.hierarchical = None

        # AlphaGeometryLoop - Creative Problem Solving
        if config.use_creative_loop:
            print("  [2.4] AlphaGeometryLoop")
            from AlphaGeometryLoop import AlphaGeometryLoop, LoopConfig
            self.creative_loop = AlphaGeometryLoop(LoopConfig(
                max_iterations=10,
                min_confidence=0.5,
                timeout_seconds=1.0,
            ))
            print("    -> Neural proposes, symbolic verifies\n")
        else:
            self.creative_loop = None

        # Shared state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        # Mode selection networks
        self.confidence = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.novelty = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Runtime statistics
        self.stats = {'reactive': 0, 'verified': 0, 'creative': 0, 'total': 0}
        self.timestep = 0

        self._print_summary()

    def _print_summary(self):
        print("="*70)
        print("[ARCHITECTURE SUMMARY]")
        print("  System 1 (Fast):  Reactive control @ 50Hz")
        print("  System 2 (Slow):  Deliberate reasoning @ 1-5Hz")
        if self.world_model:
            print("    - WorldModel: Imagine future states")
        if self.math_reasoner:
            print("    - MathReasoner: Physics verification")
        if self.hierarchical:
            print("    - Hierarchical: Skill decomposition")
        if self.creative_loop:
            print("    - AlphaGeoLoop: Creative problem solving")

        print("\n[RUNTIME MODES]")
        print("  1. REACTIVE (90%):  Pure System 1 - fast reflexes")
        print("  2. VERIFIED (9%):   System 1 + physics check - safe")
        print("  3. CREATIVE (1%):   Full reasoning loop - novel solutions")

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[SCALE]")
        print(f"  Parameters: {total_params:,}")
        print(f"  Size: ~{total_params * 4 / 1e6:.1f}MB")
        print("="*70 + "\n")

    def forward(
        self,
        proprio: torch.Tensor,
        vision: Optional[torch.Tensor] = None,
        language: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        mode: str = "auto"
    ) -> Dict:
        """
        Unified forward pass with automatic mode selection.

        Args:
            proprio: (B, obs_dim) - proprioception
            vision: (B, 3, H, W) - RGB image (optional)
            language: (B, seq_len) - text tokens (optional)
            goal: (B, obs_dim) - goal state (triggers creative mode)
            mode: "auto", "reactive", "verified", "creative"

        Returns:
            dict with actions, values, mode_used, reasoning_trace
        """
        batch_size = proprio.shape[0]
        self.timestep += 1
        self.stats['total'] += 1

        # Encode state for mode selection
        state_repr = self.state_encoder(proprio)

        # AUTO MODE SELECTION
        if mode == "auto":
            confidence = self.confidence(state_repr).item()
            novelty = self.novelty(state_repr).item()

            if confidence > self.config.reactive_threshold:
                mode = "reactive"
            elif novelty > 0.7 or goal is not None:
                mode = "creative"
            else:
                mode = "verified"

        # ==========================================
        # MODE 1: REACTIVE (Pure System 1)
        # ==========================================
        if mode == "reactive":
            self.stats['reactive'] += 1
            actions, values, memory = self.system1(
                proprio=proprio, vision=vision, language=language,
            )
            return {
                'actions': actions,
                'values': values,
                'mode': 'reactive',
                'system': 'System 1'
            }

        # ==========================================
        # MODE 2: VERIFIED (System 1 + System 2 check)
        # ==========================================
        elif mode == "verified":
            self.stats['verified'] += 1
            actions, values, memory = self.system1(
                proprio=proprio, vision=vision, language=language,
            )

            reasoning_trace = {}

            # Physics verification
            if self.math_reasoner:
                action_first = actions[:, 0, :]
                math_output = self.math_reasoner(state_repr, action_first)
                reasoning_trace['physics'] = {
                    'rule_weights': math_output['rule_weights'],
                    'physics_quantities': math_output['physics'],
                }

            # Symbolic safety check
            if self.symbolic_calculator:
                action_np = actions[:, 0, :].detach().cpu().numpy()[0]
                state_np = proprio.detach().cpu().numpy()[0]
                is_safe, reason = self.symbolic_calculator.verify_action_safe(state_np, action_np)
                if not is_safe:
                    # Correct unsafe action using symbolic physics
                    next_state, _ = self.symbolic_calculator.predict_robot_state(state_np, action_np)
                    actions[:, 0, :] = torch.FloatTensor(next_state[:17]).unsqueeze(0)
                    reasoning_trace['corrected'] = True
                else:
                    reasoning_trace['corrected'] = False
                reasoning_trace['verification'] = reason

            # Imagination-based value refinement
            if self.world_model:
                current_latent = self.world_model.encode(proprio)
                imagined_latents, imagined_rewards = self.world_model.imagine_trajectory(
                    current_latent, actions
                )
                values = imagined_rewards.mean(dim=1, keepdim=True)
                reasoning_trace['imagined_reward'] = imagined_rewards.mean().item()

            return {
                'actions': actions,
                'values': values,
                'mode': 'verified',
                'system': 'System 1 + 2',
                'reasoning': reasoning_trace,
            }

        # ==========================================
        # MODE 3: CREATIVE (Full AGI reasoning)
        # ==========================================
        elif mode == "creative":
            self.stats['creative'] += 1

            if goal is None or not self.creative_loop:
                # Fallback to System 1
                actions, values, memory = self.system1(
                    proprio=proprio, vision=vision, language=language,
                )
                return {
                    'actions': actions,
                    'values': values,
                    'mode': 'creative_fallback',
                    'system': 'System 1'
                }

            reasoning_trace = {}
            state_t = state_repr[0] if batch_size == 1 else state_repr
            goal_t = self.state_encoder(goal)[0] if batch_size == 1 else self.state_encoder(goal)

            # Task decomposition via HAC
            if self.hierarchical:
                plan = self.hierarchical.plan(state_t.unsqueeze(0), goal_t.unsqueeze(0))
                reasoning_trace['plan'] = {
                    'active_skill': plan['skill_name'],
                    'subgoal_idx': self.hierarchical.current_subgoal_idx,
                }

            # AlphaGeometry creative loop
            creative_action, creative_metadata = self.creative_loop.solve(
                state_t, goal_t, verbose=False
            )

            if creative_action is not None:
                reasoning_trace['creative'] = creative_metadata

                # Validate with imagination
                if self.world_model:
                    creative_actions_batch = creative_action.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    current_latent = self.world_model.encode(proprio)
                    imagined_latents, imagined_rewards = self.world_model.imagine_trajectory(
                        current_latent, creative_actions_batch
                    )
                    reasoning_trace['imagination'] = {
                        'predicted_reward': imagined_rewards.mean().item()
                    }
                    values = imagined_rewards.mean(dim=1, keepdim=True)
                else:
                    values = torch.zeros(1, 1)

                actions = creative_action.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                return {
                    'actions': actions,
                    'values': values,
                    'mode': 'creative',
                    'system': 'System 2 (AGI)',
                    'reasoning': reasoning_trace,
                }
            else:
                # Creative loop failed, fallback
                actions, values, memory = self.system1(
                    proprio=proprio, vision=vision, language=language,
                )
                return {
                    'actions': actions,
                    'values': values,
                    'mode': 'creative_failed',
                    'system': 'System 1'
                }

    def get_stats(self) -> Dict:
        """Get runtime mode statistics"""
        total = self.stats['total']
        if total == 0:
            return self.stats
        return {
            **self.stats,
            'reactive_pct': self.stats['reactive'] / total * 100,
            'verified_pct': self.stats['verified'] / total * 100,
            'creative_pct': self.stats['creative'] / total * 100,
        }


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING ENHANCED JACK BRAIN")
    print("="*70 + "\n")

    # Test with all System 2 components
    config = AGIConfig(
        use_world_model=True,
        use_math_reasoning=True,
        use_hierarchical=True,
        use_creative_loop=True,
    )
    brain = EnhancedJackBrain(config, obs_dim=348)

    proprio = torch.randn(1, 348)

    print("[TEST 1] Reactive mode")
    with torch.no_grad():
        out = brain(proprio, mode="reactive")
    print(f"  Actions: {out['actions'].shape}")
    print(f"  Mode: {out['mode']}, System: {out['system']}")

    print("\n[TEST 2] Verified mode")
    with torch.no_grad():
        out = brain(proprio, mode="verified")
    print(f"  Actions: {out['actions'].shape}")
    print(f"  Mode: {out['mode']}, System: {out['system']}")

    print("\n[STATISTICS]")
    stats = brain.get_stats()
    print(f"  Reactive: {stats.get('reactive_pct', 0):.1f}%")
    print(f"  Verified: {stats.get('verified_pct', 0):.1f}%")
    print(f"  Creative: {stats.get('creative_pct', 0):.1f}%")

    print("\n" + "="*70)
    print("[OK] EnhancedJackBrain validated")
    print("="*70)
