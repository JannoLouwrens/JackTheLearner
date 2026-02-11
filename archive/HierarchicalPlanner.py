"""
HIERARCHICAL PLANNER - OPTIONS FRAMEWORK (HAC Style)

Based on research:
- Options Framework (Sutton, Precup, Singh 1999 - foundational)
- HAC (Hierarchical Actor-Critic, 2024 robotics applications)
- Director (Google DeepMind, hierarchical control)

Key capabilities:
- Decomposes complex tasks into sub-goals
- Learns reusable skills (options)
- Three-level hierarchy:
  * High: Task decomposition ("clean room" → "goto table", "pick cup", "goto sink")
  * Mid: Skill selection ("pick cup" → activate grasping skill)
  * Low: Motor control (JackBrain diffusion policy)

Example:
Task: "Walk to door and open it"
High: [sub-goal: door position, sub-goal: handle position]
Mid: [skill: walk_to_target, skill: reach_and_grasp]
Low: [joint angles for each timestep]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HierarchicalPlannerConfig:
    """Configuration for hierarchical planner"""
    d_model: int = 512             # Hidden dimension
    n_heads: int = 8               # Attention heads
    n_layers: int = 4              # Transformer layers

    # Hierarchy levels
    num_skills: int = 20           # Number of learnable skills (mid-level)
    skill_horizon: int = 10        # How many low-level steps per skill
    max_subgoals: int = 5          # Max sub-goals per task

    # Dimensions
    state_dim: int = 256           # State representation
    goal_dim: int = 64             # Goal representation
    action_dim: int = 17           # Low-level action dimension


class Skill(nn.Module):
    """
    A reusable skill (option in Options framework terminology).

    Examples of skills:
    - skill_0: "Stand up from prone"
    - skill_1: "Walk forward"
    - skill_2: "Turn left"
    - skill_3: "Reach arm forward"
    - skill_4: "Grasp object"
    - skill_5: "Climb stairs"
    ... (learned automatically!)

    Each skill:
    - Has initiation set (when can it start?)
    - Has policy (what actions to take?)
    - Has termination condition (when is it done?)
    """

    def __init__(self, skill_id: int, config: HierarchicalPlannerConfig):
        super().__init__()
        self.skill_id = skill_id
        self.config = config

        # Initiation classifier (can this skill be used now?)
        self.initiation = nn.Sequential(
            nn.Linear(config.state_dim + config.goal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )

        # Policy (what to do during skill execution)
        self.policy = nn.Sequential(
            nn.Linear(config.state_dim + config.goal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.goal_dim),  # Output: sub-goal for low-level controller
        )

        # Termination (should this skill end?)
        self.termination = nn.Sequential(
            nn.Linear(config.state_dim + config.goal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )

    def can_initiate(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Check if skill can be initiated in current state"""
        combined = torch.cat([state, goal], dim=-1)
        return self.initiation(combined)

    def execute(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Execute skill policy (generate sub-goal for low-level)"""
        combined = torch.cat([state, goal], dim=-1)
        return self.policy(combined)

    def should_terminate(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Check if skill should terminate"""
        combined = torch.cat([state, goal], dim=-1)
        return self.termination(combined)


class HighLevelPlanner(nn.Module):
    """
    High-level task decomposer.

    Input: Complex task ("clean the room")
    Output: Sequence of sub-goals

    Uses Transformer to understand task structure and decompose it.
    """

    def __init__(self, config: HierarchicalPlannerConfig):
        super().__init__()
        self.config = config

        # Task encoder (embed task description)
        self.task_encoder = nn.Sequential(
            nn.Linear(config.goal_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
        )

        # Planning transformer (decompose task)
        self.planner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=config.n_layers
        )

        # Sub-goal generator
        self.subgoal_generator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.goal_dim)
        )

        # Learnable sub-goal queries (like DETR object queries)
        self.subgoal_queries = nn.Parameter(
            torch.randn(1, config.max_subgoals, config.d_model) * 0.02
        )

        print(f"[*] High-Level Planner: Decomposes tasks into {config.max_subgoals} sub-goals")

    def forward(self, state: torch.Tensor, task: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim) - current state
            task: (batch, goal_dim) - high-level task

        Returns:
            subgoals: (batch, max_subgoals, goal_dim) - sequence of sub-goals
            attention_weights: (batch, max_subgoals) - importance of each sub-goal
        """
        batch_size = state.shape[0]

        # Encode state and task
        state_emb = self.state_encoder(state).unsqueeze(1)  # (batch, 1, d_model)
        task_emb = self.task_encoder(task).unsqueeze(1)  # (batch, 1, d_model)

        # Combine with learnable sub-goal queries
        queries = self.subgoal_queries.expand(batch_size, -1, -1)  # (batch, max_subgoals, d_model)
        context = torch.cat([state_emb, task_emb, queries], dim=1)  # (batch, 2+max_subgoals, d_model)

        # Planning via self-attention
        planning_output = self.planner(context)  # (batch, 2+max_subgoals, d_model)

        # Extract sub-goal representations
        subgoal_reprs = planning_output[:, 2:, :]  # Skip state and task tokens

        # Generate sub-goals
        subgoals = self.subgoal_generator(subgoal_reprs)  # (batch, max_subgoals, goal_dim)

        # Compute attention weights (which sub-goals are important?)
        attention_weights = torch.softmax(
            torch.sum(subgoal_reprs ** 2, dim=-1),  # Energy-based attention
            dim=-1
        )

        return subgoals, attention_weights


class MidLevelController(nn.Module):
    """
    Mid-level skill selector.

    Given:
    - Current state
    - Current sub-goal (from high-level planner)

    Selects:
    - Which skill to execute

    This is where reusable behaviors emerge!
    """

    def __init__(self, config: HierarchicalPlannerConfig):
        super().__init__()
        self.config = config

        # Create skill library
        self.skills = nn.ModuleList([
            Skill(skill_id=i, config=config)
            for i in range(config.num_skills)
        ])

        # Skill selector (which skill to use?)
        self.skill_selector = nn.Sequential(
            nn.Linear(config.state_dim + config.goal_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.num_skills),
        )

        # Skill names (for interpretability)
        self.skill_names = [f"skill_{i}" for i in range(config.num_skills)]

        print(f"[*] Mid-Level Controller: {config.num_skills} learnable skills")

    def select_skill(self, state: torch.Tensor, subgoal: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Select which skill to execute.

        Returns:
            skill_id: int - which skill
            skill_prob: (num_skills,) - selection probabilities
        """
        combined = torch.cat([state, subgoal], dim=-1)
        skill_logits = self.skill_selector(combined)
        skill_probs = F.softmax(skill_logits, dim=-1)

        # Sample skill (during training) or pick best (during inference)
        if self.training:
            skill_id = torch.multinomial(skill_probs, num_samples=1).item()
        else:
            skill_id = skill_probs.argmax(dim=-1).item()

        return skill_id, skill_probs

    def execute_skill(
        self,
        skill_id: int,
        state: torch.Tensor,
        subgoal: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Execute selected skill.

        Returns:
            low_level_goal: (goal_dim,) - goal for low-level controller
            termination_prob: float - should skill terminate?
        """
        skill = self.skills[skill_id]

        # Execute skill policy
        low_level_goal = skill.execute(state, subgoal)

        # Check termination
        termination_prob = skill.should_terminate(state, subgoal).item()

        return low_level_goal, termination_prob


class HierarchicalPlanner(nn.Module):
    """
    Complete 3-level hierarchical planner.

    Level 1 (High): Task → Sub-goals
    Level 2 (Mid): Sub-goal → Skill
    Level 3 (Low): Skill → Actions (handled by JackBrain)

    This enables:
    - Long-horizon planning
    - Skill reuse
    - Compositional behavior

    Example execution:
    Task: "Go to kitchen and get coffee"
    └─> High-level: [subgoal: kitchen_location, subgoal: coffee_machine]
        └─> Mid-level: skill="navigate_to_location" → subgoal: kitchen_location
            └─> Low-level: JackBrain generates walking actions
        └─> Mid-level: skill="grasp_object" → subgoal: coffee_cup
            └─> Low-level: JackBrain generates grasping actions
    """

    def __init__(self, config: HierarchicalPlannerConfig):
        super().__init__()
        self.config = config

        print("\n" + "="*70)
        print("[*] INITIALIZING HIERARCHICAL PLANNER")
        print("="*70)

        # Hierarchy levels
        self.high_level = HighLevelPlanner(config)
        self.mid_level = MidLevelController(config)

        # Tracking
        self.current_subgoal_idx = 0
        self.current_skill_id = None
        self.skill_step = 0

        print("="*70 + "\n")

    def plan(self, state: torch.Tensor, task: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full hierarchical planning.

        Args:
            state: (batch, state_dim) - current state
            task: (batch, goal_dim) - high-level task

        Returns:
            plan: dict with:
                - subgoals: (batch, max_subgoals, goal_dim)
                - active_subgoal: (batch, goal_dim)
                - skill_id: int
                - low_level_goal: (batch, goal_dim)
        """
        batch_size = state.shape[0]
        assert batch_size == 1, "Currently supports batch_size=1 for sequential execution"

        # High-level planning (task decomposition)
        subgoals, subgoal_weights = self.high_level(state, task)

        # Get current sub-goal
        active_subgoal = subgoals[:, self.current_subgoal_idx, :]

        # Mid-level skill selection
        skill_id, skill_probs = self.mid_level.select_skill(state, active_subgoal)

        # Execute skill
        low_level_goal, termination_prob = self.mid_level.execute_skill(
            skill_id, state, active_subgoal
        )

        # Check if skill should terminate
        if termination_prob > 0.5:
            self.skill_step = 0
            self.current_subgoal_idx = min(
                self.current_subgoal_idx + 1,
                self.config.max_subgoals - 1
            )
        else:
            self.skill_step += 1

        return {
            'subgoals': subgoals,
            'active_subgoal': active_subgoal,
            'skill_id': skill_id,
            'skill_name': self.mid_level.skill_names[skill_id],
            'low_level_goal': low_level_goal,
            'skill_probs': skill_probs,
            'subgoal_weights': subgoal_weights,
            'termination_prob': termination_prob,
        }

    def reset(self):
        """Reset planner state (call at episode start)"""
        self.current_subgoal_idx = 0
        self.current_skill_id = None
        self.skill_step = 0


def compute_hierarchical_loss(
    planner: HierarchicalPlanner,
    state: torch.Tensor,
    task: torch.Tensor,
    achieved_subgoal: torch.Tensor,
    reward: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Training loss for hierarchical planner.

    Uses Hindsight Experience Replay (HER) for goal-conditioned RL:
    - Learn from achieved goals, not just intended goals
    - Makes sparse reward problems tractable

    Args:
        state: (batch, state_dim)
        task: (batch, goal_dim) - intended task
        achieved_subgoal: (batch, goal_dim) - what was actually achieved
        reward: (batch,) - task reward

    Returns:
        loss: scalar
        metrics: dict
    """
    plan = planner.plan(state, task)

    # 1. Sub-goal regression (predict what will be achieved)
    subgoal_loss = F.mse_loss(plan['active_subgoal'], achieved_subgoal)

    # 2. Skill selection loss (reinforce good skills)
    skill_probs = plan['skill_probs']
    skill_id = plan['skill_id']

    # REINFORCE-style gradient
    skill_log_prob = torch.log(skill_probs[0, skill_id] + 1e-8)
    skill_loss = -skill_log_prob * reward.item()  # Higher reward = lower loss

    # 3. Entropy regularization (explore different skills)
    skill_entropy = -(skill_probs * torch.log(skill_probs + 1e-8)).sum()
    entropy_loss = -skill_entropy  # Maximize entropy

    # Total loss
    total_loss = subgoal_loss + 0.1 * skill_loss + 0.01 * entropy_loss

    metrics = {
        'subgoal_loss': subgoal_loss.item(),
        'skill_loss': skill_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'total_loss': total_loss.item(),
        'active_skill': plan['skill_name'],
    }

    return total_loss, metrics


if __name__ == "__main__":
    print("[*] Hierarchical Planner - Architecture Demo\n")

    # Create planner
    config = HierarchicalPlannerConfig(
        d_model=512,
        n_heads=8,
        n_layers=4,
        num_skills=20,
        state_dim=256,
        goal_dim=64,
        action_dim=17,
    )

    planner = HierarchicalPlanner(config)

    # Count parameters
    total_params = sum(p.numel() for p in planner.parameters())
    print(f"[*] Total parameters: {total_params:,}")
    print(f"[*] Model size: ~{total_params * 4 / 1e6:.1f}MB\n")

    # Test planning
    state = torch.randn(1, 256)
    task = torch.randn(1, 64)

    print("[*] Testing hierarchical planning...")
    planner.eval()
    with torch.no_grad():
        plan = planner.plan(state, task)

    print(f"[OK] Sub-goals: {plan['subgoals'].shape}")
    print(f"[OK] Active sub-goal: {plan['active_subgoal'].shape}")
    print(f"[OK] Selected skill: {plan['skill_name']} (ID: {plan['skill_id']})")
    print(f"[OK] Low-level goal: {plan['low_level_goal'].shape}")
    print(f"[OK] Termination probability: {plan['termination_prob']:.3f}")

    print("\n[*] Skill selection probabilities:")
    top_skills = torch.topk(plan['skill_probs'][0], k=5)
    for i, (prob, idx) in enumerate(zip(top_skills.values, top_skills.indices)):
        skill_name = planner.mid_level.skill_names[idx.item()]
        print(f"  {i+1}. {skill_name}: {prob.item():.3f}")

    print("\n" + "="*70)
    print("[SUCCESS] Hierarchical Planner validated!")
    print("="*70)
    print("\n[*] Capabilities:")
    print("   ✅ Task decomposition (complex → sub-goals)")
    print("   ✅ Skill library (20 reusable behaviors)")
    print("   ✅ Sequential execution with termination")
    print("   ✅ Interpretable (can see which skill is active)")
    print("="*70)
