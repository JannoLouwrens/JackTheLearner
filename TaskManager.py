"""
TASK MANAGER - Persistent Goal-Directed Behavior

Bridges high-level commands ("make tea") to per-frame actions across
thousands of timesteps. Jack can now pursue multi-step goals.

Architecture (hybrid of proven approaches):
- SayCan (Google 2022): Affordance-grounded skill selection
- Inner Monologue (Google 2022): Closed-loop feedback and replanning
- Voyager (NVIDIA 2023): Skill library with self-verification
- Behavior Trees: Tick-based execution with SUCCESS/FAILURE/RUNNING

Execution model:
    User says "make tea"
        → TaskManager.set_task("make tea")
        → LLM/Planner decomposes into subtasks:
            [walk_to_kitchen, find_cup, pick_up_cup, boil_water, pour, bring_back]
        → Each frame: TaskManager.tick(state)
            → Feeds current subtask goal to brain
            → Checks TaskCompletionHead for done
            → Advances to next subtask when done
            → Replans on failure (Inner Monologue feedback)

Author: Janno Louwrens
"""

import time
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

import torch

try:
    from EmotionalState import EventType
except ImportError:
    EventType = None


# ==============================================================================
# SUBTASK STATUS
# ==============================================================================

class SubtaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


# ==============================================================================
# SUBTASK
# ==============================================================================

@dataclass
class Subtask:
    """A single step in a multi-step task."""
    description: str
    status: SubtaskStatus = SubtaskStatus.PENDING
    frames_running: int = 0
    max_frames: int = 3000          # ~60 seconds at 50Hz
    completion_threshold: float = 0.8
    attempts: int = 0
    max_attempts: int = 3
    started_at: float = 0.0
    completed_at: float = 0.0


# ==============================================================================
# TASK DECOMPOSITION TEMPLATES
# ==============================================================================

# Known task decompositions (used when LLM is unavailable)
KNOWN_TASKS = {
    "make tea": [
        "walk to the kitchen",
        "find the cup",
        "pick up the cup",
        "walk to the kettle",
        "put the cup down",
        "wait for water to boil",
        "pour water into cup",
        "pick up the cup",
        "walk back to start",
    ],
    "make coffee": [
        "walk to the kitchen",
        "find the cup",
        "pick up the cup",
        "walk to the coffee machine",
        "place cup under spout",
        "press the button",
        "wait for coffee",
        "pick up the cup",
        "walk back to start",
    ],
    "fetch the ball": [
        "find the ball",
        "walk to the ball",
        "pick up the ball",
        "walk back to start",
    ],
    "clean up": [
        "find the cup",
        "pick up the cup",
        "walk to the table",
        "put the cup on the table",
        "find the ball",
        "pick up the ball",
        "walk to the shelf",
        "put the ball on the shelf",
    ],
    "explore the room": [
        "walk to the table",
        "look around",
        "walk to the shelf",
        "look around",
        "walk to the door",
        "look around",
        "walk back to start",
    ],
}

# Simple commands that are a single subtask
SIMPLE_COMMANDS = {
    "walk forward": ["walk forward"],
    "walk backward": ["walk backward"],
    "turn left": ["turn left"],
    "turn right": ["turn right"],
    "stop": ["stop"],
    "sit down": ["sit down"],
    "stand up": ["stand up"],
    "wave": ["wave"],
    "look around": ["look around"],
}


# ==============================================================================
# TASK MANAGER
# ==============================================================================

class TaskManager:
    """
    Persistent task manager that bridges high-level commands to per-frame actions.

    Called every frame from VirtualWorld. Persists goals across thousands of frames.
    Detects subtask completion, handles failure with replanning.

    Usage:
        tm = TaskManager(brain)
        tm.set_task("make tea")

        # Every frame in game loop:
        result = tm.tick(state, current_time)
        action = result['action']
    """

    def __init__(self, brain):
        """
        Args:
            brain: UnifiedBrain instance (has act_with_mood, TaskCompletionHead,
                   HierarchicalPlanner, InnerMonologue, emotional_state)
        """
        self.brain = brain

        # Current task state
        self.task_description: str = ""
        self.subtasks: List[Subtask] = []
        self.current_idx: int = 0
        self.active: bool = False

        # History for Inner Monologue feedback
        self.completed_steps: List[str] = []
        self.task_history: List[Dict] = []  # Past tasks with outcomes

        # Stuck detection (Inner Monologue replanning trigger)
        self.stuck_frames: int = 0
        self.stuck_threshold: int = 500  # ~10 seconds without progress
        self.last_task_done_prob: float = 0.0

        # Stats
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def set_task(self, command: str) -> str:
        """
        Set a new high-level task. Decomposes into subtasks.

        Args:
            command: Natural language command ("make tea", "fetch the ball")

        Returns:
            Response string ("Okay, I'll make tea. First I'll walk to the kitchen.")
        """
        self.task_description = command.strip()
        self.subtasks = self._decompose(command)
        self.current_idx = 0
        self.active = True
        self.completed_steps = []
        self.stuck_frames = 0

        # Emotional response: excitement about new task
        if self.brain.emotional_state is not None and EventType is not None:
            self.brain.emotional_state.update(
                event_type=EventType.GOAL_ACHIEVED,  # New goal = motivating
                reward=0.3,
                user_interaction=True,
                dt=0.1,
            )

        # Generate response
        if self.subtasks:
            first = self.subtasks[0].description
            response = f"Okay, I'll {command}. First, I'll {first}."
        else:
            response = f"I'll try to {command}."
            self.active = False

        # Store in memory
        if hasattr(self.brain, 'memory') and self.brain.memory is not None:
            self.brain.memory.add(f"Task started: {command}", importance=0.7)

        # Inner monologue
        if hasattr(self.brain, 'inner_monologue') and self.brain.inner_monologue is not None:
            self.brain.inner_monologue._record(
                f"New task: {command}. Plan: {[s.description for s in self.subtasks]}",
                thought_type="plan",
            )

        return response

    def tick(self, state: torch.Tensor, current_time: float = 0.0,
             vision: 'torch.Tensor' = None, touch: 'torch.Tensor' = None,
             audio: 'torch.Tensor' = None) -> Dict:
        """
        Called EVERY FRAME. Returns action dict.

        If a task is active, feeds the current subtask goal to the brain.
        If no task, returns default autonomous behavior.

        Args:
            state: Current observation tensor [1, obs_dim]
            current_time: Monotonic time for timing
            vision: Optional eye camera image [1, 3, 224, 224]
            touch: Optional contact force data [1, 10]

        Returns:
            Dict with 'action', 'mood', 'task_info', etc.
        """
        if not self.active or not self.subtasks:
            # No active task - return default behavior
            return self._idle_tick(state, current_time, vision=vision, touch=touch, audio=audio)

        # Get current subtask
        subtask = self.subtasks[self.current_idx]

        if subtask.status == SubtaskStatus.PENDING:
            subtask.status = SubtaskStatus.RUNNING
            subtask.started_at = current_time
            subtask.frames_running = 0

        subtask.frames_running += 1

        # ── Generate goal-conditioned action ──
        # Feed subtask description + all senses to brain
        if hasattr(self.brain, 'act_with_mood'):
            result = self.brain.act_with_mood(
                state,
                language=subtask.description,
                vision=vision,
                touch=touch,
                audio=audio,
                current_time=current_time,
                is_idle=False,
            )
        else:
            with torch.no_grad():
                output = self.brain(state)
            result = {
                'action': output['actions'][:, 0, :],
                'task_done': output.get('task_done', torch.tensor([0.0])),
            }

        # ── Check completion (SayCan + Inner Monologue feedback) ──
        task_done_prob = 0.0
        if 'task_done' in result:
            td = result['task_done']
            if isinstance(td, torch.Tensor):
                task_done_prob = td.item() if td.numel() == 1 else td.mean().item()
            else:
                task_done_prob = float(td)

        # Also check full_output if available
        if 'full_output' in result and 'task_done' in result['full_output']:
            td2 = result['full_output']['task_done']
            if isinstance(td2, torch.Tensor):
                task_done_prob = max(task_done_prob, td2.mean().item())

        # ── Stuck detection ──
        progress = abs(task_done_prob - self.last_task_done_prob)
        self.last_task_done_prob = task_done_prob

        if progress < 0.01:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        # ── Subtask completion ──
        if task_done_prob > subtask.completion_threshold:
            self._on_subtask_success(subtask, current_time)
            self._advance(current_time)

        # ── Timeout ──
        elif subtask.frames_running > subtask.max_frames:
            self._on_subtask_timeout(subtask, current_time)

        # ── Stuck → replan ──
        elif self.stuck_frames > self.stuck_threshold:
            self._on_stuck(subtask, current_time)

        # ── Add task info to result ──
        result['task_info'] = {
            'active': self.active,
            'task': self.task_description,
            'current_subtask': subtask.description if self.active else None,
            'subtask_idx': self.current_idx,
            'total_subtasks': len(self.subtasks),
            'task_done_prob': task_done_prob,
            'frames_running': subtask.frames_running,
            'status': subtask.status.value,
        }

        return result

    def cancel(self) -> str:
        """Cancel the current task."""
        if self.active:
            task = self.task_description
            self.active = False
            self.task_description = ""
            self.subtasks = []
            return f"Okay, I'll stop trying to {task}."
        return "I wasn't doing anything."

    @property
    def has_task(self) -> bool:
        return self.active

    @property
    def current_subtask(self) -> Optional[str]:
        if self.active and self.current_idx < len(self.subtasks):
            return self.subtasks[self.current_idx].description
        return None

    @property
    def progress(self) -> float:
        """0.0 to 1.0 progress through the task."""
        if not self.subtasks:
            return 0.0
        return self.current_idx / len(self.subtasks)

    # ─────────────────────────────────────────────────────────────────────
    # TASK DECOMPOSITION
    # ─────────────────────────────────────────────────────────────────────

    def _decompose(self, command: str) -> List[Subtask]:
        """
        Decompose a high-level command into subtasks.

        Priority:
        1. Check known task templates
        2. Check simple commands
        3. Use API LLM to decompose (if available)
        4. Use HierarchicalPlanner (if trained)
        5. Fallback: treat entire command as single subtask
        """
        cmd_lower = command.lower().strip()

        # 1. Known task templates
        for key, steps in KNOWN_TASKS.items():
            if key in cmd_lower:
                return [Subtask(description=s) for s in steps]

        # 2. Simple single-step commands
        for key, steps in SIMPLE_COMMANDS.items():
            if key in cmd_lower:
                return [Subtask(description=s, max_frames=500) for s in steps]

        # 3. API LLM decomposition
        subtasks = self._decompose_with_llm(command)
        if subtasks:
            return subtasks

        # 4. Keyword extraction fallback
        subtasks = self._decompose_keywords(command)
        if subtasks:
            return subtasks

        # 5. Single subtask fallback
        return [Subtask(description=command)]

    def _decompose_with_llm(self, command: str) -> Optional[List[Subtask]]:
        """Use API LLM to decompose command into subtasks."""
        api_llm = getattr(self.brain, 'api_llm', None)
        if api_llm is None or not api_llm.available:
            return None

        prompt = (
            f"Break this robot command into simple sequential steps. "
            f"Each step should be a single physical action like 'walk to X', "
            f"'pick up X', 'put X on Y', 'look around', 'wait', etc.\n\n"
            f"Command: {command}\n\n"
            f"Return ONLY a numbered list of steps, nothing else."
        )

        try:
            response = api_llm.generate(
                system_prompt="You decompose robot tasks into simple steps.",
                user_message=prompt,
                max_tokens=200,
            )

            if not response:
                return None

            # Parse numbered list
            steps = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering: "1. walk to kitchen" → "walk to kitchen"
                if line and line[0].isdigit():
                    line = line.lstrip('0123456789.-) ').strip()
                if line and len(line) > 2:
                    steps.append(Subtask(description=line))

            return steps if steps else None

        except Exception:
            return None

    def _decompose_keywords(self, command: str) -> Optional[List[Subtask]]:
        """Extract subtasks from keywords in the command."""
        cmd = command.lower()
        steps = []

        # "go/walk to X" pattern
        for prefix in ["go to", "walk to", "move to", "head to"]:
            if prefix in cmd:
                target = cmd.split(prefix)[-1].strip().rstrip('.')
                steps.append(Subtask(description=f"walk to the {target}"))

        # "pick up / grab X" pattern
        for prefix in ["pick up", "grab", "get", "take"]:
            if prefix in cmd:
                target = cmd.split(prefix)[-1].strip().rstrip('.')
                target = target.replace("the ", "").strip()
                if target:
                    steps.append(Subtask(description=f"walk to the {target}"))
                    steps.append(Subtask(description=f"pick up the {target}"))

        # "bring X to Y" pattern
        if "bring" in cmd:
            parts = cmd.split("bring")[-1].strip()
            if " to " in parts:
                obj, dest = parts.split(" to ", 1)
                obj = obj.strip().replace("the ", "").replace("me ", "").strip()
                dest = dest.strip().rstrip('.')
                steps = [
                    Subtask(description=f"find the {obj}"),
                    Subtask(description=f"walk to the {obj}"),
                    Subtask(description=f"pick up the {obj}"),
                    Subtask(description=f"walk to {dest}"),
                    Subtask(description=f"put down the {obj}"),
                ]

        # "kick X" pattern
        if "kick" in cmd:
            target = cmd.split("kick")[-1].strip().rstrip('.')
            target = target.replace("the ", "").strip()
            steps = [
                Subtask(description=f"walk to the {target}"),
                Subtask(description=f"kick the {target}"),
            ]

        return steps if steps else None

    # ─────────────────────────────────────────────────────────────────────
    # SUBTASK LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────

    def _idle_tick(self, state: torch.Tensor, current_time: float,
                   vision: 'torch.Tensor' = None, touch: 'torch.Tensor' = None,
                   audio: 'torch.Tensor' = None) -> Dict:
        """Default behavior when no task is active."""
        if hasattr(self.brain, 'act_with_mood'):
            return self.brain.act_with_mood(state, vision=vision, touch=touch,
                                             audio=audio, current_time=current_time, is_idle=True)
        with torch.no_grad():
            output = self.brain(state)
        return {'action': output['actions'][:, 0, :], 'task_info': {'active': False}}

    def _advance(self, current_time: float):
        """Move to the next subtask or complete the task."""
        self.current_idx += 1
        self.stuck_frames = 0
        self.last_task_done_prob = 0.0

        if self.current_idx >= len(self.subtasks):
            # Task complete!
            self._on_task_complete(current_time)
        else:
            next_sub = self.subtasks[self.current_idx]
            # Inner monologue: narrate progress
            if hasattr(self.brain, 'inner_monologue') and self.brain.inner_monologue is not None:
                self.brain.inner_monologue._record(
                    f"Done with '{self.completed_steps[-1]}'. Now: '{next_sub.description}'",
                    thought_type="plan",
                )

    def _on_subtask_success(self, subtask: Subtask, current_time: float):
        """Called when a subtask completes successfully."""
        subtask.status = SubtaskStatus.SUCCESS
        subtask.completed_at = current_time
        self.completed_steps.append(subtask.description)

        # Emotional response: satisfaction
        if self.brain.emotional_state is not None and EventType is not None:
            self.brain.emotional_state.update(
                event_type=EventType.TASK_SUCCESS,
                reward=0.5,
                dt=0.1,
            )

        # Memory
        if hasattr(self.brain, 'memory') and self.brain.memory is not None:
            self.brain.memory.add(
                f"Completed step: {subtask.description} (task: {self.task_description})",
                importance=0.4,
            )

    def _on_subtask_timeout(self, subtask: Subtask, current_time: float):
        """Called when a subtask times out."""
        subtask.attempts += 1

        if subtask.attempts < subtask.max_attempts:
            # Retry
            subtask.frames_running = 0
            subtask.status = SubtaskStatus.RUNNING
            self.stuck_frames = 0

            if hasattr(self.brain, 'inner_monologue') and self.brain.inner_monologue is not None:
                self.brain.inner_monologue._record(
                    f"Hmm, '{subtask.description}' is taking too long. Let me try again.",
                    thought_type="reflection",
                )

            # Frustration
            if self.brain.emotional_state is not None and EventType is not None:
                self.brain.emotional_state.update(
                    event_type=EventType.TASK_FAILURE,
                    reward=-0.3,
                    dt=0.1,
                )
        else:
            # Give up on this subtask, skip it
            subtask.status = SubtaskStatus.FAILURE
            self.completed_steps.append(f"[FAILED] {subtask.description}")

            if hasattr(self.brain, 'inner_monologue') and self.brain.inner_monologue is not None:
                self.brain.inner_monologue._record(
                    f"I can't seem to '{subtask.description}'. I'll skip it and move on.",
                    thought_type="reflection",
                )

            self._advance(current_time)

    def _on_stuck(self, subtask: Subtask, current_time: float):
        """Called when no progress is made for many frames."""
        self.stuck_frames = 0  # Reset counter

        if hasattr(self.brain, 'inner_monologue') and self.brain.inner_monologue is not None:
            self.brain.inner_monologue._record(
                f"I seem stuck on '{subtask.description}'. Let me think about this...",
                thought_type="appraisal",
            )

        # Try replanning via LLM
        new_subtasks = self._replan(subtask)
        if new_subtasks:
            # Replace remaining subtasks with new plan
            self.subtasks = (
                self.subtasks[:self.current_idx] + new_subtasks
            )
            if hasattr(self.brain, 'inner_monologue') and self.brain.inner_monologue is not None:
                self.brain.inner_monologue._record(
                    f"New plan: {[s.description for s in new_subtasks]}",
                    thought_type="plan",
                )
        else:
            # No replan available, increment attempt counter
            subtask.attempts += 1
            if subtask.attempts >= subtask.max_attempts:
                self._on_subtask_timeout(subtask, current_time)

    def _replan(self, failed_subtask: Subtask) -> Optional[List[Subtask]]:
        """Ask LLM to suggest an alternative approach."""
        api_llm = getattr(self.brain, 'api_llm', None)
        if api_llm is None or not api_llm.available:
            return None

        prompt = (
            f"I'm a robot trying to '{self.task_description}'.\n"
            f"I've completed: {self.completed_steps}\n"
            f"I'm stuck on: '{failed_subtask.description}'\n"
            f"Suggest alternative steps to continue. "
            f"Return ONLY a numbered list of simple physical actions."
        )

        try:
            response = api_llm.generate(
                system_prompt="You help a robot replan when it's stuck.",
                user_message=prompt,
                max_tokens=200,
            )

            if not response:
                return None

            steps = []
            for line in response.strip().split('\n'):
                line = line.strip().lstrip('0123456789.-) ').strip()
                if line and len(line) > 2:
                    steps.append(Subtask(description=line))

            return steps if steps else None

        except Exception:
            return None

    def _on_task_complete(self, current_time: float):
        """Called when all subtasks are done."""
        self.active = False
        self.tasks_completed += 1

        # Strong positive emotion
        if self.brain.emotional_state is not None and EventType is not None:
            self.brain.emotional_state.update(
                event_type=EventType.GOAL_ACHIEVED,
                reward=1.0,
                dt=0.1,
            )

        # Memory
        if hasattr(self.brain, 'memory') and self.brain.memory is not None:
            self.brain.memory.add(
                f"Completed task: {self.task_description}!",
                importance=0.9,
            )

        # Inner monologue: satisfaction
        if hasattr(self.brain, 'inner_monologue') and self.brain.inner_monologue is not None:
            self.brain.inner_monologue._record(
                f"I did it! I finished '{self.task_description}'. That feels good.",
                thought_type="reflection",
            )

        # Task history
        self.task_history.append({
            'task': self.task_description,
            'subtasks': [s.description for s in self.subtasks],
            'success': True,
            'time': current_time,
        })

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────

    def get_state(self) -> Dict:
        """Get serializable state for saving."""
        return {
            'task_description': self.task_description,
            'subtasks': [
                {
                    'description': s.description,
                    'status': s.status.value,
                    'frames_running': s.frames_running,
                    'attempts': s.attempts,
                }
                for s in self.subtasks
            ],
            'current_idx': self.current_idx,
            'active': self.active,
            'completed_steps': self.completed_steps,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
        }

    def load_state(self, state: Dict):
        """Restore state from save."""
        self.task_description = state.get('task_description', '')
        self.current_idx = state.get('current_idx', 0)
        self.active = state.get('active', False)
        self.completed_steps = state.get('completed_steps', [])
        self.tasks_completed = state.get('tasks_completed', 0)
        self.tasks_failed = state.get('tasks_failed', 0)

        self.subtasks = []
        for sd in state.get('subtasks', []):
            sub = Subtask(description=sd['description'])
            sub.status = SubtaskStatus(sd.get('status', 'pending'))
            sub.frames_running = sd.get('frames_running', 0)
            sub.attempts = sd.get('attempts', 0)
            self.subtasks.append(sub)


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("TaskManager Test")
    print("=" * 50)

    # Test decomposition without brain
    class MockBrain:
        emotional_state = None
        memory = None
        inner_monologue = None
        api_llm = None

        def act_with_mood(self, state, **kwargs):
            return {
                'action': torch.randn(1, 17),
                'task_done': torch.tensor([0.1]),
                'mood': {},
            }

    brain = MockBrain()
    tm = TaskManager(brain)

    # Test known task
    response = tm.set_task("make tea")
    print(f"Task: make tea")
    print(f"Response: {response}")
    print(f"Subtasks: {[s.description for s in tm.subtasks]}")
    assert len(tm.subtasks) == 9, f"Expected 9 subtasks, got {len(tm.subtasks)}"

    # Test simple command
    tm2 = TaskManager(brain)
    tm2.set_task("walk forward")
    assert len(tm2.subtasks) == 1

    # Test keyword extraction
    tm3 = TaskManager(brain)
    tm3.set_task("bring the ball to the table")
    print(f"\nTask: bring the ball to the table")
    print(f"Subtasks: {[s.description for s in tm3.subtasks]}")
    assert len(tm3.subtasks) >= 3

    # Test tick
    state = torch.randn(1, 256)
    result = tm.tick(state, current_time=0.0)
    assert 'action' in result
    assert 'task_info' in result
    assert result['task_info']['active']
    print(f"\nTick result: subtask='{result['task_info']['current_subtask']}' done_prob={result['task_info']['task_done_prob']:.2f}")

    # Test persistence
    saved = tm.get_state()
    tm_new = TaskManager(brain)
    tm_new.load_state(saved)
    assert tm_new.active == tm.active
    assert len(tm_new.subtasks) == len(tm.subtasks)
    print(f"\nPersistence: round-trip OK ({len(tm_new.subtasks)} subtasks)")

    # Test cancel
    cancel_msg = tm.cancel()
    assert not tm.active
    print(f"\nCancel: '{cancel_msg}'")

    print("\n=== ALL TESTS PASSED ===")
