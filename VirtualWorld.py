"""
VIRTUAL WORLD - Jack's Home (Game Loop)

A thin runtime layer that drives the UnifiedBrain. The brain is the SINGLE
source of truth for all companion state (emotional state, personality,
inner monologue, movement-mood coupling). VirtualWorld does NOT own or
duplicate any of these modules -- it accesses them via self.brain.

Responsibilities:
- MuJoCo simulation with offscreen rendering (mujoco.Renderer)
- PyGame window for display and user input
- Driving the brain's act_with_mood() loop each frame
- Reading emotional state FROM the brain for UI display
- Triggering emotional events on the brain's emotional state
- Auto-save via the Persistence module

Research backing:
- MuJoCo Renderer: Offscreen rendering via mujoco.Renderer (Todorov et al. 2012)
- Dual System: GR00T N1 / Figure Helix timing (System 2 at 9Hz, System 1 at 50Hz)
- Game AI: Generative Agents (Park et al. 2023) for autonomous behavior

Usage:
    python VirtualWorld.py                    # Launch with default scene
    python VirtualWorld.py --scene room       # Launch with room scene
    python VirtualWorld.py --text-only        # Text-only mode (no rendering)

Author: Janno Louwrens
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── Optional dependency imports ─────────────────────────────────────────────
# Each is guarded so the module loads cleanly even on minimal installs.

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    mujoco = None  # type: ignore
    HAS_MUJOCO = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    pygame = None  # type: ignore
    HAS_PYGAME = False

# ─── Project imports ─────────────────────────────────────────────────────────

from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig

try:
    from EmotionalState import EmotionalState, EmotionalConfig, EventType
except ImportError:
    EmotionalState = None  # type: ignore
    EmotionalConfig = None  # type: ignore
    EventType = None  # type: ignore

try:
    from Personality import Personality, PersonalityConfig
except ImportError:
    Personality = None  # type: ignore
    PersonalityConfig = None  # type: ignore

try:
    from InnerMonologue import InnerMonologue, MonologueConfig
except ImportError:
    InnerMonologue = None  # type: ignore
    MonologueConfig = None  # type: ignore

try:
    from Persistence import CompanionPersistence, SaveConfig
except ImportError:
    CompanionPersistence = None  # type: ignore
    SaveConfig = None  # type: ignore

try:
    from TaskManager import TaskManager
except ImportError:
    TaskManager = None

try:
    from AudioListener import AudioListener, AudioConfig
except ImportError:
    AudioListener = None
    AudioConfig = None


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def apply_action(mj_data, mj_model, action) -> None:
    """Write an action to the actuators, or raise. Never silently adapt.

    The previous implementation was:

        n_act = min(len(self.current_action), self.mj_model.nu)
        self.mj_data.ctrl[:n_act] = self.current_action[:n_act]

    which TRUNCATES. A policy emitting 16 values for a 17-actuator humanoid drove
    16 joints and left the last holding a stale command; one emitting 40 used the
    first 17 and discarded the rest. Neither raised. Every locomotion number
    produced that way would have been meaningless and nothing in the logs would
    have said so.

    Plain assignment is not enough either: `ctrl[:] = np.zeros(1)` is accepted by
    NumPy broadcasting and drives EVERY joint with one value (measured, ladder
    spec T0.06). So the width is checked explicitly before the write.
    """
    import numpy as np

    a = np.asarray(action, dtype=np.float64).reshape(-1)
    nu = int(mj_model.nu)
    if a.size != nu:
        raise ValueError(
            f"action width {a.size} != model.nu {nu}. Refusing to write: "
            "truncating or broadcasting here drives the wrong joints silently. "
            "Fix the policy's action_dim or the model, do not pad."
        )
    if not np.all(np.isfinite(a)):
        raise ValueError("action contains NaN or Inf; refusing to write to ctrl")
    mj_data.ctrl[:] = a


@dataclass

class WorldConfig:
    """Configuration for the virtual world and game loop.

    Attributes:
        width: Render window width in pixels.
        height: Render window height in pixels.
        target_fps: Target frames per second for the main loop.
                    50 Hz matches the System 1 action generation frequency
                    from GR00T N1 / Figure Helix architecture.
        physics_substeps: Number of MuJoCo sub-steps per frame for stability.
        auto_save_interval: Seconds between automatic state saves.
        idle_threshold: Seconds of user inactivity before autonomous behavior.
        camera_speed: Degrees per frame for arrow-key camera rotation.
        scene_xml: Path to the MuJoCo scene XML file.
        save_dir: Directory for save files.
    """
    width: int = 800
    height: int = 600
    target_fps: int = 50
    physics_substeps: int = 5
    auto_save_interval: float = 300.0
    idle_threshold: float = 30.0
    camera_speed: float = 1.5
    scene_xml: Optional[str] = None
    save_dir: str = "saves"


# =============================================================================
# DEFAULT SCENE (inline MuJoCo XML for a minimal room)
# =============================================================================

_DEFAULT_SCENE_XML = """
<mujoco model="jack_room">
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="50"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.8 0.8 0.85" rgb2="0.4 0.42 0.44" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8"
              reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
  </asset>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>

    <!-- Ground plane -->
    <geom name="floor" type="plane" size="10 10 0.1" material="grid_mat"/>

    <!-- Walls (simple boxes) -->
    <body name="wall_north" pos="0 5 1">
      <geom type="box" size="5 0.1 1" rgba="0.7 0.7 0.75 1"/>
    </body>
    <body name="wall_south" pos="0 -5 1">
      <geom type="box" size="5 0.1 1" rgba="0.7 0.7 0.75 1"/>
    </body>
    <body name="wall_east" pos="5 0 1">
      <geom type="box" size="0.1 5 1" rgba="0.7 0.7 0.75 1"/>
    </body>
    <body name="wall_west" pos="-5 0 1">
      <geom type="box" size="0.1 5 1" rgba="0.7 0.7 0.75 1"/>
    </body>

    <!-- Interactive objects -->
    <body name="ball" pos="2 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.15" rgba="0.9 0.2 0.2 1" mass="0.5"/>
    </body>
    <body name="cube" pos="-1 1.5 0.2">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.15" rgba="0.2 0.7 0.3 1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


# =============================================================================
# SCENE CATALOG
# =============================================================================

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

SCENE_CATALOG: Dict[str, str] = {
    "room": os.path.join(_ASSET_DIR, "manipulation_scene.xml"),
    "humanoid": os.path.join(_ASSET_DIR, "humanoid_full.xml"),
    "terrain": os.path.join(_ASSET_DIR, "humanoid_terrain.xml"),
}


# =============================================================================
# COLOR PALETTE (PyGame drawing)
# =============================================================================

class _Colors:
    """Centralized color definitions so UI drawing is consistent."""
    BLACK       = (0, 0, 0)
    WHITE       = (255, 255, 255)
    DARK_GRAY   = (40, 40, 45)
    MID_GRAY    = (80, 80, 90)
    LIGHT_GRAY  = (180, 180, 190)
    PANEL_BG    = (25, 28, 35, 200)

    # Mood-bar colors
    PLEASURE_POS = (60, 200, 80)
    PLEASURE_NEG = (200, 60, 60)
    AROUSAL_HIGH = (230, 200, 40)
    AROUSAL_LOW  = (80, 80, 140)
    DOMINANCE_HI = (50, 120, 220)
    DOMINANCE_LO = (130, 60, 180)

    # Chat
    USER_MSG     = (120, 180, 255)
    JACK_MSG     = (100, 230, 140)
    SYSTEM_MSG   = (200, 200, 110)

    # Thought bubble
    BUBBLE_BG    = (255, 255, 245, 220)
    BUBBLE_BORDER = (180, 170, 150)
    THOUGHT_TEXT = (60, 55, 50)


# =============================================================================
# VIRTUAL WORLD (Full GUI)
# =============================================================================

class VirtualWorld:
    """Thin runtime layer that drives the UnifiedBrain in a MuJoCo world via PyGame.

    CORE PRINCIPLE: The brain is the SINGLE source of truth for all state.
    VirtualWorld does NOT own emotional_state, personality, inner_monologue,
    or movement_mood. It accesses them through self.brain.

    Responsibilities:
    - MuJoCo simulation with offscreen rendering (mujoco.Renderer)
    - PyGame window for display and user input
    - Calling brain.act_with_mood() each frame for action generation
    - Reading emotional state FROM the brain for UI display (mood bars)
    - Triggering emotional events on the brain's emotional state
    - Driving brain.inner_monologue for autonomous thoughts
    - Auto-save via the Persistence module

    Rendering pipeline (MuJoCo -> PyGame):
        1. mujoco.Renderer renders scene offscreen to an internal buffer
        2. renderer.render() produces an RGB numpy array (H, W, 3)
        3. The array is transposed from (H, W, 3) to (W, H, 3) for PyGame
        4. pygame.surfarray.blit_array writes the array to the display surface
        5. UI overlays (mood bars, chat, thoughts, status) are drawn on top

    Research:
        MuJoCo offscreen rendering follows Todorov et al. (2012).
        The dual-system brain tick follows GR00T N1 / Figure Helix timing.
        Autonomous idle behavior follows Park et al. (2023) Generative Agents.
    """

    # ──────────────────────────────────────────────────────────────────────
    # CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        brain: UnifiedBrain,
        config: Optional[WorldConfig] = None,
    ) -> None:
        self.brain = brain
        self.config = config or WorldConfig()
        self.running = False
        self.paused = False

        # NOTE: All companion modules (emotional_state, personality,
        # inner_monologue, movement_mood) live on the brain.
        # Access them via self.brain.emotional_state, etc.
        # VirtualWorld does NOT own any of these.

        # ── Persistence ───────────────────────────────────────────────────
        self.persistence: Optional[Any] = None
        if CompanionPersistence is not None and SaveConfig is not None:
            save_cfg = SaveConfig(
                save_dir=self.config.save_dir,
                auto_save_interval=self.config.auto_save_interval,
            )
            self.persistence = CompanionPersistence(save_cfg)

        # ── Task Manager (persistent multi-step goal execution) ──────────
        self.task_manager = None
        if TaskManager is not None:
            self.task_manager = TaskManager(self.brain)
            print("[TaskManager] Ready for multi-step commands")

        # ── Audio Listener (microphone input) ────────────────────────────
        self.audio_listener = None
        if AudioListener is not None:
            self.audio_listener = AudioListener(
                config=AudioConfig() if AudioConfig else None,
                on_transcription=self._on_voice_input,
            )
            if self.audio_listener.available:
                self.audio_listener.start()
                print("[AudioListener] Microphone active - Jack can hear you")
            else:
                print("[AudioListener] No microphone available - use text chat")

        # ── MuJoCo setup ─────────────────────────────────────────────────
        self.mj_model = None
        self.mj_data = None
        self.mj_renderer = None
        self._init_mujoco()

        # ── PyGame setup ──────────────────────────────────────────────────
        self.screen = None
        self.clock = None
        self.font_small = None
        self.font_medium = None
        self.font_large = None
        self._init_pygame()

        # ── Camera state ──────────────────────────────────────────────────
        self.camera_azimuth: float = 180.0
        self.camera_elevation: float = -20.0
        self.camera_distance: float = 4.0
        self.camera_lookat: np.ndarray = np.array([0.0, 0.0, 0.8])

        # ── Chat state ───────────────────────────────────────────────────
        self.chat_history: deque = deque(maxlen=50)
        self.chat_input: str = ""
        self.chat_active: bool = False  # True when user is typing

        # ── Timing ────────────────────────────────────────────────────────
        self.frame_count: int = 0
        self.start_time: float = time.monotonic()
        self.last_user_input_time: float = time.monotonic()
        self.last_brain_tick: float = time.monotonic()
        self.last_system2_tick: float = 0.0

        # ── Current state (observation from MuJoCo) ──────────────────────
        self.current_obs: Optional[np.ndarray] = None
        self.current_action: Optional[np.ndarray] = None
        self.current_task: str = "Idle"
        self.active_skill: str = "none"

        # ── Thought bubble state ──────────────────────────────────────────
        self.current_thought: str = ""
        self.thought_display_until: float = 0.0

        print("[VirtualWorld] Initialization complete.")
        self._log_chat("SYSTEM", "Welcome to Jack's World! Press Enter to chat, Esc to quit.")

    # ──────────────────────────────────────────────────────────────────────
    # INITIALIZATION HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _init_mujoco(self) -> None:
        """Initialize MuJoCo model, data, and offscreen renderer.

        If a scene_xml path is configured and exists, it is loaded.
        Otherwise the built-in default room scene is used (inline XML).
        Falls back gracefully if MuJoCo is not installed.
        """
        if not HAS_MUJOCO:
            print("[VirtualWorld] MuJoCo not available -- physics disabled.")
            return

        scene_xml = self.config.scene_xml

        try:
            if scene_xml is not None and os.path.isfile(scene_xml):
                print(f"[VirtualWorld] Loading scene: {scene_xml}")
                self.mj_model = mujoco.MjModel.from_xml_path(scene_xml)
            else:
                if scene_xml is not None:
                    print(f"[VirtualWorld] Scene file not found: {scene_xml}")
                    print("[VirtualWorld] Falling back to default room scene.")
                self.mj_model = mujoco.MjModel.from_xml_string(_DEFAULT_SCENE_XML)

            self.mj_data = mujoco.MjData(self.mj_model)
            self.mj_renderer = mujoco.Renderer(
                self.mj_model,
                height=self.config.height,
                width=self.config.width,
            )
            print(f"[VirtualWorld] MuJoCo ready: {self.mj_model.nq} qpos, "
                  f"{self.mj_model.nv} qvel, {self.mj_model.nu} actuators")

        except Exception as exc:
            print(f"[VirtualWorld] MuJoCo init failed: {exc}")
            self.mj_model = None
            self.mj_data = None
            self.mj_renderer = None

    def _init_pygame(self) -> None:
        """Initialize PyGame display, clock, and fonts.

        Falls back gracefully if PyGame is not installed.
        """
        if not HAS_PYGAME:
            print("[VirtualWorld] PyGame not available -- GUI disabled.")
            return

        try:
            pygame.init()
            pygame.display.set_caption("Jack's World - Virtual Companion")
            self.screen = pygame.display.set_mode(
                (self.config.width, self.config.height),
            )
            self.clock = pygame.time.Clock()

            # Fonts
            self.font_small = pygame.font.SysFont("consolas", 13)
            self.font_medium = pygame.font.SysFont("consolas", 16)
            self.font_large = pygame.font.SysFont("consolas", 22, bold=True)

            print(f"[VirtualWorld] PyGame ready: {self.config.width}x{self.config.height}")

        except Exception as exc:
            print(f"[VirtualWorld] PyGame init failed: {exc}")
            self.screen = None
            self.clock = None

    # ──────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main game loop running at ~50 fps (configurable).

        Loop structure per frame:
        1. _handle_events()    -- process PyGame input events
        2. _update_brain()     -- tick brain's dual system architecture
        3. _update_emotional() -- decay and event-based mood updates
        4. _update_autonomous() -- inner monologue and exploration when idle
        5. _step_physics()     -- advance MuJoCo simulation
        6. _render()           -- draw MuJoCo frame + UI overlay
        7. _auto_save_tick()   -- periodic state persistence

        The loop exits when self.running is set to False (Esc key or window close).
        """
        if self.screen is None:
            print("[VirtualWorld] No display available. Use TextOnlyWorld instead.")
            return

        self.running = True
        self.start_time = time.monotonic()
        self.last_user_input_time = time.monotonic()
        print("[VirtualWorld] Entering main loop. Press Esc to quit.")

        try:
            while self.running:
                now = time.monotonic()

                self._handle_events()

                if not self.paused:
                    self._update_brain(now)
                    self._update_emotional(now)
                    self._update_autonomous(now)
                    self._step_physics()

                self._render()
                self._auto_save_tick(now)

                self.frame_count += 1
                if self.clock is not None:
                    self.clock.tick(self.config.target_fps)

        except KeyboardInterrupt:
            print("\n[VirtualWorld] Interrupted by user.")
        finally:
            self.cleanup()

    # ──────────────────────────────────────────────────────────────────────
    # EVENT HANDLING
    # ──────────────────────────────────────────────────────────────────────

    def _handle_events(self) -> None:
        """Process all pending PyGame events.

        Handles:
        - QUIT: window close button
        - KEYDOWN: keyboard input for chat, camera, and control
        - Special keys: Esc (quit), Space (pause), Enter (submit chat),
          Backspace (edit chat), arrow keys (camera rotation)
        """
        if pygame is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

    def _handle_keydown(self, event: Any) -> None:
        """Process a single KEYDOWN event."""
        key = event.key

        # ── Global controls (always active) ───────────────────────────────
        if key == pygame.K_ESCAPE:
            if self.chat_active:
                # First press: cancel chat input
                self.chat_active = False
                self.chat_input = ""
            else:
                # Second press: quit
                self.running = False
            return

        if key == pygame.K_SPACE and not self.chat_active:
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            self._log_chat("SYSTEM", f"Simulation {status}")
            return

        # ── Chat mode toggle ──────────────────────────────────────────────
        if key == pygame.K_RETURN:
            if self.chat_active:
                # Submit the message
                if self.chat_input.strip():
                    self._send_chat(self.chat_input.strip())
                self.chat_input = ""
                self.chat_active = False
            else:
                # Activate chat input
                self.chat_active = True
            self.last_user_input_time = time.monotonic()
            return

        # ── Chat text input ───────────────────────────────────────────────
        if self.chat_active:
            if key == pygame.K_BACKSPACE:
                self.chat_input = self.chat_input[:-1]
            elif key == pygame.K_TAB:
                pass  # Ignore tab in chat
            else:
                char = event.unicode
                if char and len(self.chat_input) < 200:
                    self.chat_input += char
            self.last_user_input_time = time.monotonic()
            return

        # ── Camera controls (only when not chatting) ──────────────────────
        speed = self.config.camera_speed
        if key == pygame.K_LEFT:
            self.camera_azimuth -= speed * 5
        elif key == pygame.K_RIGHT:
            self.camera_azimuth += speed * 5
        elif key == pygame.K_UP:
            self.camera_elevation = max(-89.0, self.camera_elevation - speed * 3)
        elif key == pygame.K_DOWN:
            self.camera_elevation = min(89.0, self.camera_elevation + speed * 3)
        elif key == pygame.K_PAGEUP:
            self.camera_distance = max(1.0, self.camera_distance - 0.3)
        elif key == pygame.K_PAGEDOWN:
            self.camera_distance = min(20.0, self.camera_distance + 0.3)

        self.last_user_input_time = time.monotonic()

    # ──────────────────────────────────────────────────────────────────────
    # BRAIN UPDATE
    # ──────────────────────────────────────────────────────────────────────

    def _update_brain(self, now: float) -> None:
        """Tick the brain via TaskManager.

        If a task is active, the TaskManager feeds the current subtask goal
        to the brain and checks for completion. If no task, the brain
        generates autonomous/idle behavior.

        The TaskManager handles:
        - Multi-step task execution (SayCan-style skill chaining)
        - Subtask completion detection (TaskCompletionHead)
        - Failure handling and replanning (Inner Monologue)
        """
        if torch is None:
            return

        try:
            obs = self._get_observation_tensor()
            if obs is None:
                return

            # ── Senses ──
            eye_image = self._get_eye_image()    # Vision: egocentric camera on head
            touch_data = self._get_touch_data()   # Touch: contact forces on body

            # Audio: speech-to-text (NOT raw streaming into transformer)
            # Research consensus (pi0, GR00T N1): speech → text → language path
            # Raw audio is only used for discrete events (loud sounds, etc.)
            audio_data = None  # Only set for significant audio events
            if self.audio_listener is not None:
                # Check for completed speech → becomes text command
                voice_text = self.audio_listener.get_transcription()
                if voice_text:
                    self._send_chat(voice_text)  # Same path as typed text

                # Check for loud ambient sounds (startle/awareness)
                if self.audio_listener.is_speaking():
                    # Someone is currently talking - get audio for awareness
                    audio_data = self.audio_listener.get_recent_audio(duration=0.5)

            # Route through TaskManager (handles both active tasks and idle)
            with torch.no_grad():
                if hasattr(self, 'task_manager') and self.task_manager is not None:
                    result = self.task_manager.tick(obs, current_time=now,
                                                    vision=eye_image, touch=touch_data,
                                                    audio=audio_data)
                else:
                    result = self.brain.act_with_mood(
                        state=obs,
                        vision=eye_image,
                        touch=touch_data,
                        audio=audio_data,
                        current_time=now,
                        is_idle=self._is_idle(),
                    )

            # Extract action
            action = result.get("action")
            if action is not None:
                if isinstance(action, torch.Tensor):
                    self.current_action = action.cpu().numpy().flatten()
                else:
                    self.current_action = action

            # Update task display
            task_info = result.get("task_info", {})
            if task_info.get("active"):
                self.current_task = f"{task_info.get('current_subtask', '')} ({task_info.get('subtask_idx', 0)+1}/{task_info.get('total_subtasks', 0)})"
            elif hasattr(self, 'task_manager') and self.task_manager is not None and not self.task_manager.active:
                self.current_task = ""

            # Track System 2 timing
            if result.get("system2_ran", False):
                self.last_system2_tick = now

        except Exception as exc:
            logger.debug("Brain tick error: %s", exc)

    def _get_eye_image(self) -> Optional[torch.Tensor]:
        """Render what Jack sees from his head-mounted eye camera.

        Returns [1, 3, 224, 224] float tensor normalized to [0, 1],
        or None if rendering is unavailable.

        The 'eye' camera is attached to Jack's head in the MuJoCo XML.
        It moves with his head, giving him egocentric vision.
        """
        if self.mj_renderer is None or self.mj_model is None or self.mj_data is None:
            return None

        if not HAS_MUJOCO:
            return None

        try:
            # Find the eye camera ID
            eye_cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
            if eye_cam_id < 0:
                return None

            # Render from eye camera (lower res for speed)
            self.mj_renderer.update_scene(self.mj_data, camera=eye_cam_id)
            rgb = self.mj_renderer.render()  # (H, W, 3) uint8

            # Convert to tensor: [1, 3, H, W] float [0, 1]
            img = torch.from_numpy(rgb.copy()).float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

            # Resize to 224x224 if needed (standard vision input size)
            if img.shape[2] != 224 or img.shape[3] != 224:
                img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

            return img

        except Exception as exc:
            logger.debug("Eye render error: %s", exc)
            return None

    def _on_voice_input(self, text: str):
        """Callback from AudioListener when speech is transcribed.
        Called from the audio background thread - just store for the game loop to pick up."""
        # The game loop checks get_transcription() each frame
        pass  # The transcription is already stored in the listener

    def _get_audio_data(self) -> Optional[torch.Tensor]:
        """Get recent audio waveform for the AudioEncoder.

        Returns [1, samples] tensor of last 1 second of audio at 16kHz,
        or None if no microphone.

        This gives Jack ambient sound awareness beyond just speech.
        """
        if self.audio_listener is None or not self.audio_listener.available:
            return None
        return self.audio_listener.get_recent_audio(duration=1.0)

    def _get_touch_data(self) -> Optional[torch.Tensor]:
        """Read contact forces from MuJoCo as touch sensor data.

        Returns [1, 10] tensor of contact force magnitudes for key body parts:
        [left_foot, right_foot, left_hand, right_hand, torso,
         left_knee, right_knee, left_elbow, right_elbow, head]

        These tell Jack what he's touching and how hard.
        """
        if self.mj_data is None or self.mj_model is None or not HAS_MUJOCO:
            return None

        try:
            # cfrc_ext: external contact forces on each body [nbody, 6]
            # First 3 are torque, last 3 are force
            cfrc = self.mj_data.cfrc_ext  # (nbody, 6)

            # Extract force magnitudes for key bodies
            # Body indices depend on the XML structure - use names if available
            touch = torch.zeros(1, 10)
            body_names = ['left_foot', 'right_foot', 'left_hand', 'right_hand',
                         'torso', 'left_shin', 'right_shin',
                         'left_lower_arm', 'right_lower_arm', 'head']

            for i, name in enumerate(body_names):
                try:
                    body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                    if body_id >= 0 and body_id < cfrc.shape[0]:
                        force = cfrc[body_id, 3:6]  # Linear force components
                        touch[0, i] = float(np.linalg.norm(force))
                except Exception:
                    pass

            # Normalize to [0, 1] range (typical forces 0-100N)
            touch = torch.clamp(touch / 100.0, 0.0, 1.0)
            return touch

        except Exception as exc:
            logger.debug("Touch read error: %s", exc)
            return None

    def _is_idle(self) -> bool:
        """Return True if no user command within idle_threshold seconds."""
        return (time.monotonic() - self.last_user_input_time) > self.config.idle_threshold

    # ──────────────────────────────────────────────────────────────────────
    # EMOTIONAL STATE UPDATE
    # ──────────────────────────────────────────────────────────────────────

    def _update_emotional(self, now: float) -> None:
        """Tick the emotional state -- decay toward baseline and process events.

        The EmotionalState module implements the ALMA three-layer model:
        - Personality baseline (stable)
        - Mood decay (slow, exponential toward baseline)
        - Emotion events (fast, triggered by specific occurrences)

        Called once per frame. The dt argument ensures time-invariant decay
        regardless of frame rate fluctuations.
        """
        if self.brain.emotional_state is None or EventType is None:
            return

        try:
            # Compute actual elapsed time for accurate decay
            if not hasattr(self, '_last_emotional_update'):
                self._last_emotional_update = now
            dt = max(0.001, min(now - self._last_emotional_update, 1.0))
            self._last_emotional_update = now

            # Check for boredom (no user input for a while)
            idle_time = now - self.last_user_input_time
            event = None
            if idle_time > 60.0:
                # Occasional boredom ticks when idle for over a minute
                if self.frame_count % (self.config.target_fps * 10) == 0:
                    event = EventType.BOREDOM_TICK

            self.brain.emotional_state.update(event_type=event, dt=dt)

        except Exception as exc:
            logger.debug("Emotional update error: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # AUTONOMOUS BEHAVIOR
    # ──────────────────────────────────────────────────────────────────────

    def _update_autonomous(self, now: float) -> None:
        """Generate autonomous behavior when the user is idle.

        Following Park et al. (2023) Generative Agents:
        - If idle for > idle_threshold seconds, trigger exploration
        - If inner_monologue cooldown has elapsed, generate a thought

        The inner monologue uses the InnerMonologue module which produces
        template-based thoughts (fallback) or LLM-generated thoughts when
        an LLM backend is available.
        """
        idle_time = now - self.last_user_input_time

        # ── Inner monologue ───────────────────────────────────────────────
        if self.brain.inner_monologue is not None:
            try:
                if self.brain.inner_monologue.should_think(now):
                    # Gather context for thought generation
                    personality_prompt = ""
                    if self.brain.personality is not None:
                        personality_prompt = self.brain.personality.get_inner_monologue_prompt(
                            mood_dict=self._get_mood_dict(),
                            current_goal=self.current_task,
                        )

                    mood_dict = self._get_mood_dict()
                    recent_memories = self._get_recent_memories()

                    thought = self.brain.inner_monologue.think(
                        personality_prompt=personality_prompt,
                        mood_dict=mood_dict,
                        recent_memories=recent_memories,
                        current_goal=self.current_task,
                    )

                    self.current_thought = thought
                    self.thought_display_until = now + 8.0  # Show for 8 seconds

                    # Store thought in brain's long-term memory
                    if thought and hasattr(self.brain, 'memory') and self.brain.memory is not None:
                        self.brain.memory.add(f"[Thought] {thought}", importance=0.5)

                    # Thoughts can trigger emotional events
                    if thought and self.brain.emotional_state is not None and EventType is not None:
                        thought_lower = thought.lower()
                        if 'curious' in thought_lower or 'wonder' in thought_lower:
                            self.brain.emotional_state.update(
                                event_type=EventType.NOVELTY, reward=0.1, dt=0.1
                            )
                        elif 'frustrated' in thought_lower or 'stuck' in thought_lower:
                            self.brain.emotional_state.update(
                                event_type=EventType.TASK_FAILURE, reward=-0.1, dt=0.1
                            )

            except Exception as exc:
                logger.debug("Inner monologue error: %s", exc)

        # ── Idle exploration trigger ──────────────────────────────────────
        if idle_time > self.config.idle_threshold:
            # Periodically announce exploration intent
            if self.frame_count % (self.config.target_fps * 30) == 0:
                self._log_chat("Jack", "Hmm, I think I'll explore a bit...")
                self.current_task = "Exploring"

    # ──────────────────────────────────────────────────────────────────────
    # PHYSICS STEP
    # ──────────────────────────────────────────────────────────────────────

    def _step_physics(self) -> None:
        """Advance the MuJoCo simulation by one frame.

        Applies the current action (if any) to the actuators, then steps
        the simulation for physics_substeps iterations. Each sub-step uses
        the model's internal timestep (default 5ms), so at 5 sub-steps per
        frame and 50 fps we get 250 physics steps/second.
        """
        if self.mj_model is None or self.mj_data is None:
            return

        try:
            # Apply action to actuators. Exact width or refuse — see apply_action.
            if self.current_action is not None and self.mj_model.nu > 0:
                apply_action(self.mj_data, self.mj_model, self.current_action)

            # Step physics
            for _ in range(self.config.physics_substeps):
                mujoco.mj_step(self.mj_model, self.mj_data)

        except Exception as exc:
            logger.debug("Physics step error: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # RENDERING
    # ──────────────────────────────────────────────────────────────────────

    def _render(self) -> None:
        """Render the current frame: MuJoCo scene + UI overlay.

        Rendering pipeline:
        1. Update MuJoCo renderer camera from current camera state
        2. Render the scene offscreen into an RGB numpy array
        3. Transpose (H,W,3) -> (W,H,3) for pygame.surfarray.blit_array
        4. Blit the array onto the PyGame display surface
        5. Draw UI overlays (mood bars, chat, thought bubble, status)
        6. Flip the display buffer
        """
        if self.screen is None or pygame is None:
            return

        # ── Step 1-4: Render MuJoCo scene to surface ─────────────────────
        if self.mj_renderer is not None and self.mj_data is not None:
            try:
                # _update_mujoco_camera calls update_scene with camera params
                self._update_mujoco_camera()
                pixels = self.mj_renderer.render()  # (H, W, 3) uint8

                # PyGame surfarray expects (W, H, 3) -- transpose first two axes
                surface_array = np.transpose(pixels, (1, 0, 2)).copy()
                pygame.surfarray.blit_array(self.screen, surface_array)

            except Exception as exc:
                # Fallback: fill with dark background
                self.screen.fill(_Colors.DARK_GRAY)
                logger.debug("Render error: %s", exc)
        else:
            # No MuJoCo: plain background
            self.screen.fill(_Colors.DARK_GRAY)

        # ── Step 5: UI overlay ────────────────────────────────────────────
        self._render_ui(self.screen)

        # ── Step 6: Flip ──────────────────────────────────────────────────
        pygame.display.flip()

    def _update_mujoco_camera(self) -> None:
        """Sync the MuJoCo renderer camera with the current camera state."""
        if self.mj_renderer is None:
            return
        # The mujoco.Renderer exposes a scene camera through update_scene's
        # camera parameter. We configure a mujoco camera struct.
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = self.camera_azimuth
        cam.elevation = self.camera_elevation
        cam.distance = self.camera_distance
        cam.lookat[:] = self.camera_lookat
        self.mj_renderer.update_scene(self.mj_data, camera=cam)

    # ──────────────────────────────────────────────────────────────────────
    # UI RENDERING
    # ──────────────────────────────────────────────────────────────────────

    def _render_ui(self, surface: Any) -> None:
        """Draw all UI overlay elements on top of the rendered scene.

        Layout:
        - Top-left: Status panel (FPS, task, mood, skill)
        - Top-right: Mood bars (Pleasure, Arousal, Dominance)
        - Bottom: Chat history + input box
        - Center-top: Thought bubble (when a thought is active)
        - Bottom-right: Controls hint
        """
        w, h = self.config.width, self.config.height

        # Status panel (top-left)
        self._render_status(surface, 10, 10)

        # Mood bars (top-right)
        self._render_mood_bars(surface, w - 210, 10)

        # Chat panel (bottom)
        chat_h = 160
        self._render_chat(surface, 10, h - chat_h - 10, w - 20, chat_h)

        # Thought bubble (center-top area)
        now = time.monotonic()
        if self.current_thought and now < self.thought_display_until:
            self._render_thought_bubble(surface, w // 2, 80)

        # Pause indicator
        if self.paused:
            self._render_pause_overlay(surface)

        # Controls hint (bottom-right)
        self._render_controls_hint(surface, w - 200, h - chat_h - 40)

    def _render_status(self, surface: Any, x: int, y: int) -> None:
        """Draw the status panel: FPS, current task, mood, active skill."""
        if self.font_small is None:
            return

        # Background panel
        panel_w, panel_h = 220, 110
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((25, 28, 35, 180))
        surface.blit(panel, (x, y))

        # FPS
        if self.clock is not None:
            fps = self.clock.get_fps()
        else:
            fps = 0.0
        lines = [
            f"FPS: {fps:.0f}",
            f"Task: {self.current_task}",
        ]

        # Dominant mood
        mood_label = self._get_dominant_mood()
        lines.append(f"Mood: {mood_label}")

        # Active skill
        lines.append(f"Skill: {self.active_skill}")

        # Time running
        elapsed = time.monotonic() - self.start_time
        m, s = divmod(int(elapsed), 60)
        h_val, m = divmod(m, 60)
        lines.append(f"Time: {h_val:02d}:{m:02d}:{s:02d}")

        for i, line in enumerate(lines):
            text_surf = self.font_small.render(line, True, _Colors.LIGHT_GRAY)
            surface.blit(text_surf, (x + 8, y + 6 + i * 20))

    def _render_mood_bars(self, surface: Any, x: int, y: int) -> None:
        """Draw three horizontal mood bars for PAD dimensions.

        Bar layout:
        - Pleasure (green when positive / red when negative)
        - Arousal  (yellow when high / dark blue when low)
        - Dominance (blue when high / purple when low)

        Each bar is centered: the midpoint represents 0.0, and the bar
        extends left (negative) or right (positive) from center.
        """
        if self.font_small is None:
            return

        pad = self._get_pad_values()
        bar_w = 180
        bar_h = 16
        spacing = 28

        # Background panel
        panel_h = spacing * 3 + 20
        panel = pygame.Surface((bar_w + 20, panel_h), pygame.SRCALPHA)
        panel.fill((25, 28, 35, 180))
        surface.blit(panel, (x - 5, y - 5))

        dims = [
            ("P", pad[0], _Colors.PLEASURE_POS, _Colors.PLEASURE_NEG),
            ("A", pad[1], _Colors.AROUSAL_HIGH, _Colors.AROUSAL_LOW),
            ("D", pad[2], _Colors.DOMINANCE_HI, _Colors.DOMINANCE_LO),
        ]

        for i, (label, value, color_pos, color_neg) in enumerate(dims):
            by = y + i * spacing

            # Label
            label_surf = self.font_small.render(f"{label}:", True, _Colors.LIGHT_GRAY)
            surface.blit(label_surf, (x, by))

            # Bar background (dark)
            bar_x = x + 22
            pygame.draw.rect(surface, _Colors.MID_GRAY, (bar_x, by + 2, bar_w, bar_h))

            # Center line
            center_x = bar_x + bar_w // 2
            pygame.draw.line(surface, _Colors.LIGHT_GRAY,
                             (center_x, by + 1), (center_x, by + bar_h + 2), 1)

            # Value bar
            clamped = max(-1.0, min(1.0, value))
            fill_w = int(abs(clamped) * (bar_w // 2))
            color = color_pos if clamped >= 0 else color_neg

            if clamped >= 0:
                pygame.draw.rect(surface, color,
                                 (center_x, by + 2, fill_w, bar_h))
            else:
                pygame.draw.rect(surface, color,
                                 (center_x - fill_w, by + 2, fill_w, bar_h))

            # Value text
            val_text = self.font_small.render(f"{value:+.2f}", True, _Colors.WHITE)
            surface.blit(val_text, (bar_x + bar_w + 4, by))

    def _render_chat(self, surface: Any, x: int, y: int, w: int, h: int) -> None:
        """Draw the chat history panel with input box at the bottom.

        The panel shows the most recent chat messages (scrolled to bottom)
        and a text input box that activates when the user presses Enter.
        """
        if self.font_small is None or self.font_medium is None:
            return

        # Background panel
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((25, 28, 35, 180))
        surface.blit(panel, (x, y))

        # ── Chat history ──────────────────────────────────────────────────
        msg_y = y + 5
        line_h = 18
        max_visible = (h - 35) // line_h

        visible_msgs = list(self.chat_history)[-max_visible:]
        for sender, text, _ts in visible_msgs:
            if sender == "You":
                color = _Colors.USER_MSG
                prefix = "You: "
            elif sender == "SYSTEM":
                color = _Colors.SYSTEM_MSG
                prefix = "[SYS] "
            else:
                color = _Colors.JACK_MSG
                prefix = "Jack: "

            # Truncate long messages for display
            display_text = prefix + text
            if len(display_text) > 90:
                display_text = display_text[:87] + "..."

            text_surf = self.font_small.render(display_text, True, color)
            surface.blit(text_surf, (x + 8, msg_y))
            msg_y += line_h

        # ── Input box ─────────────────────────────────────────────────────
        input_y = y + h - 28
        input_rect = pygame.Rect(x + 4, input_y, w - 8, 24)

        # Draw input box
        border_color = _Colors.USER_MSG if self.chat_active else _Colors.MID_GRAY
        pygame.draw.rect(surface, (35, 38, 45), input_rect)
        pygame.draw.rect(surface, border_color, input_rect, 1)

        # Draw input text or placeholder
        if self.chat_active:
            display = self.chat_input + "|"
            text_color = _Colors.WHITE
        elif self.chat_input:
            display = self.chat_input
            text_color = _Colors.LIGHT_GRAY
        else:
            display = "Press Enter to chat..."
            text_color = _Colors.MID_GRAY

        input_surf = self.font_small.render(display, True, text_color)
        surface.blit(input_surf, (x + 10, input_y + 4))

    def _render_thought_bubble(self, surface: Any, cx: int, y: int) -> None:
        """Draw Jack's current inner thought as a thought bubble.

        The bubble is drawn centered at (cx, y) with rounded corners and
        a small tail pointing downward to indicate it originates from Jack.
        """
        if self.font_medium is None or not self.current_thought:
            return

        # Measure text
        text_surf = self.font_medium.render(self.current_thought, True, _Colors.THOUGHT_TEXT)
        tw, th = text_surf.get_size()

        # Bubble dimensions
        pad = 14
        bw = tw + pad * 2
        bh = th + pad * 2
        bx = cx - bw // 2
        by = y

        # Draw bubble background (with alpha)
        bubble = pygame.Surface((bw, bh), pygame.SRCALPHA)
        bubble.fill((255, 255, 245, 220))
        surface.blit(bubble, (bx, by))

        # Border
        pygame.draw.rect(surface, _Colors.BUBBLE_BORDER, (bx, by, bw, bh), 2)

        # Tail (three small circles below, like a thought bubble)
        tail_x = cx
        tail_y = by + bh
        for i, (r, dy) in enumerate([(4, 6), (3, 14), (2, 20)]):
            pygame.draw.circle(surface, (255, 255, 245), (tail_x, tail_y + dy), r)
            pygame.draw.circle(surface, _Colors.BUBBLE_BORDER,
                               (tail_x, tail_y + dy), r, 1)

        # Text
        surface.blit(text_surf, (bx + pad, by + pad))

    def _render_pause_overlay(self, surface: Any) -> None:
        """Draw a semi-transparent pause overlay."""
        if self.font_large is None:
            return

        overlay = pygame.Surface(
            (self.config.width, self.config.height), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 100))
        surface.blit(overlay, (0, 0))

        text = self.font_large.render("PAUSED (Space to resume)", True, _Colors.WHITE)
        tw, th = text.get_size()
        surface.blit(text, (
            self.config.width // 2 - tw // 2,
            self.config.height // 2 - th // 2,
        ))

    def _render_controls_hint(self, surface: Any, x: int, y: int) -> None:
        """Draw a small controls reference."""
        if self.font_small is None:
            return

        hints = [
            "Enter: Chat | Esc: Quit",
            "Space: Pause | Arrows: Camera",
        ]
        for i, hint in enumerate(hints):
            text_surf = self.font_small.render(hint, True, _Colors.MID_GRAY)
            surface.blit(text_surf, (x, y + i * 16))

    # ──────────────────────────────────────────────────────────────────────
    # CHAT
    # ──────────────────────────────────────────────────────────────────────

    def _send_chat(self, message: str) -> None:
        """Process a user chat message.

        1. Log the user's message to chat history
        2. Trigger a USER_CHAT emotional event
        3. Pass the message through brain.chat() for response generation
        4. Log Jack's response to chat history
        5. Store the conversation in memory

        If the brain does not have a chat method, a fallback response is used.
        """
        now = time.monotonic()
        self._log_chat("You", message)
        self.last_user_input_time = now

        # Emotional event: user is chatting (update brain's emotional state)
        if self.brain.emotional_state is not None and EventType is not None:
            try:
                self.brain.emotional_state.update(
                    event_type=EventType.USER_CHAT,
                    user_interaction=0.8,
                )
            except Exception:
                pass

        # Route through TaskManager for multi-step commands, chat for conversation
        response = ""
        msg_lower = message.lower().strip()
        try:
            # Check if this is a task command (multi-step action)
            is_task = any(kw in msg_lower for kw in [
                "make ", "fetch ", "bring ", "clean ", "go make", "go get",
                "pick up", "put ", "carry ", "take ", "move ",
            ])

            if is_task and hasattr(self, 'task_manager') and self.task_manager is not None:
                # Route to TaskManager for persistent multi-step execution
                response = self.task_manager.set_task(message)
            elif msg_lower in ("stop", "cancel", "nevermind", "never mind"):
                # Cancel current task
                if hasattr(self, 'task_manager') and self.task_manager is not None:
                    response = self.task_manager.cancel()
                else:
                    response = "I wasn't doing anything."
            elif hasattr(self.brain, "chat"):
                # Regular conversation / simple commands
                obs = self._get_observation_tensor()
                response = self.brain.chat(message, state=obs, speak=False)
            elif hasattr(self.brain, "interact"):
                obs = self._get_observation_tensor()
                if obs is not None:
                    result = self.brain.interact(obs, message, speak=False)
                    response = result.get("response", "")
        except Exception as exc:
            logger.debug("Chat error: %s", exc)

        if not response:
            response = "I heard you. I'm still learning to respond better!"

        self._log_chat("Jack", response)

        # Store in memory
        try:
            if hasattr(self.brain, "remember"):
                self.brain.remember(f"User said: {message}")
                self.brain.remember(f"I replied: {response}")
        except Exception:
            pass

    def _log_chat(self, sender: str, text: str) -> None:
        """Append a message to the chat history."""
        self.chat_history.append((sender, text, time.monotonic()))

    # ──────────────────────────────────────────────────────────────────────
    # AUTO-SAVE
    # ──────────────────────────────────────────────────────────────────────

    def _auto_save_tick(self, now: float) -> None:
        """Check and perform periodic auto-save."""
        if self.persistence is None:
            return

        try:
            world_state = self._collect_world_state()
            self.persistence.auto_save_tick(
                brain=self.brain,
                world_state=world_state,
                current_time=now,
            )
        except Exception as exc:
            logger.debug("Auto-save error: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ──────────────────────────────────────────────────────────────────────

    def _get_observation_tensor(self) -> Optional[Any]:
        """Build an observation tensor from the current MuJoCo state.

        Returns a torch.Tensor of shape [1, obs_dim] suitable for
        brain.act_with_mood().

        If MuJoCo is not available, returns a zero tensor.

        NOTE: The truncate/pad approach targets obs_dim (default 256) which
        must match the training observation format. In production, the brain
        should handle obs projection internally; for now we pad here.
        """
        if torch is None:
            return None

        obs_dim = self.brain.config.obs_dim

        if self.mj_data is not None:
            # Concatenate qpos and qvel, pad/truncate to obs_dim
            # This must match the observation format the brain was trained with
            qpos = self.mj_data.qpos.copy()
            qvel = self.mj_data.qvel.copy()
            raw = np.concatenate([qpos, qvel])

            if len(raw) >= obs_dim:
                obs = raw[:obs_dim]
            else:
                obs = np.zeros(obs_dim, dtype=np.float32)
                obs[:len(raw)] = raw

            self.current_obs = obs.astype(np.float32)
        else:
            self.current_obs = np.zeros(obs_dim, dtype=np.float32)

        return torch.tensor(self.current_obs, dtype=torch.float32).unsqueeze(0)

    def _get_pad_values(self) -> Tuple[float, float, float]:
        """Return current PAD values (Pleasure, Arousal, Dominance) from brain."""
        if self.brain.emotional_state is not None:
            try:
                d = self.brain.emotional_state.get_mood_dict()
                return (d["pleasure"], d["arousal"], d["dominance"])
            except Exception:
                pass
        return (0.0, 0.0, 0.0)

    def _get_dominant_mood(self) -> str:
        """Return the dominant mood label string from brain."""
        if self.brain.emotional_state is not None:
            try:
                return self.brain.emotional_state.get_dominant_mood()
            except Exception:
                pass
        return "Calm"

    def _get_mood_dict(self) -> Dict[str, float]:
        """Return full mood dictionary from brain's emotional state."""
        if self.brain.emotional_state is not None:
            try:
                return self.brain.emotional_state.get_mood_dict()
            except Exception:
                pass
        return {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}

    def _get_recent_memories(self) -> List[str]:
        """Retrieve recent memory texts for inner monologue context."""
        try:
            if hasattr(self.brain, "memory") and self.brain.memory is not None:
                raw = getattr(self.brain.memory, "memories", [])
                return [m.get("text", "") for m in raw[-5:]]
        except Exception:
            pass
        return []

    def _collect_world_state(self) -> Dict[str, Any]:
        """Collect current world state for persistence."""
        state: Dict[str, Any] = {
            "frame_count": self.frame_count,
            "elapsed_time": time.monotonic() - self.start_time,
            "current_task": self.current_task,
            "camera": {
                "azimuth": self.camera_azimuth,
                "elevation": self.camera_elevation,
                "distance": self.camera_distance,
            },
        }

        if self.mj_data is not None:
            state["jack_position"] = self.mj_data.qpos[:3].tolist()

        return state

    # ──────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Save state, close renderer, and quit PyGame.

        Called automatically when the main loop exits (normal or exception).
        Also callable manually for graceful shutdown.
        """
        print("[VirtualWorld] Cleaning up...")

        # Stop audio listener
        if self.audio_listener is not None:
            try:
                self.audio_listener.stop()
            except Exception:
                pass

        # Final save
        if self.persistence is not None:
            try:
                world_state = self._collect_world_state()
                path = self.persistence.save_all(
                    self.brain, world_state=world_state
                )
                print(f"[VirtualWorld] Final save: {path}")
            except Exception as exc:
                print(f"[VirtualWorld] Save failed: {exc}")

        # Close MuJoCo renderer
        if self.mj_renderer is not None:
            try:
                self.mj_renderer.close()
            except Exception:
                pass
            self.mj_renderer = None

        # Quit PyGame
        if HAS_PYGAME and pygame.get_init():
            try:
                pygame.quit()
            except Exception:
                pass

        print("[VirtualWorld] Shutdown complete.")


# =============================================================================
# TEXT-ONLY WORLD (Fallback)
# =============================================================================

class TextOnlyWorld:
    """Fallback interaction mode when no GUI is available.

    Provides a simple text-based interface using standard input/output.
    All companion modules (emotional state, personality, inner monologue)
    are accessed through self.brain -- the brain is the single source of truth.

    This mode is useful for:
    - Remote SSH sessions (no display)
    - Systems without PyGame or MuJoCo installed
    - Quick testing and debugging
    - Environments like Google Colab (headless)
    """

    def __init__(
        self,
        brain: UnifiedBrain,
        save_dir: str = "saves",
    ) -> None:
        self.brain = brain
        # NOTE: All companion modules (emotional_state, personality,
        # inner_monologue) live on the brain. Access via self.brain.*

        # Persistence
        self.persistence: Optional[Any] = None
        if CompanionPersistence is not None and SaveConfig is not None:
            self.persistence = CompanionPersistence(
                SaveConfig(save_dir=save_dir)
            )

        # Task Manager
        self.task_manager = TaskManager(brain) if TaskManager is not None else None

        self.running = False
        self.step_count = 0

    def run(self) -> None:
        """Simple REPL loop for text-based interaction.

        Commands:
        - /quit or /exit: Exit the loop
        - /mood: Display current emotional state
        - /think: Trigger an inner monologue thought
        - /save: Manual save
        - /status: Display brain status
        - Any other text: chat with Jack
        """
        self.running = True
        print()
        print("=" * 60)
        print("  JACK'S WORLD - Text Mode")
        print("=" * 60)
        print()
        print("  Type a message to chat with Jack.")
        print("  Commands: /quit /mood /think /save /status")
        print()

        while self.running:
            try:
                user_input = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break

            elif user_input.lower() == "/mood":
                self._show_mood()
                continue

            elif user_input.lower() == "/think":
                self._trigger_thought()
                continue

            elif user_input.lower() == "/save":
                self._manual_save()
                continue

            elif user_input.lower() == "/status":
                self._show_status()
                continue

            # Chat with Jack
            self._chat(user_input)
            self.step_count += 1

            # Periodic auto-save
            if self.persistence is not None and self.step_count % 20 == 0:
                try:
                    self.persistence.auto_save_tick(
                        brain=self.brain,
                        world_state={"mode": "text", "step": self.step_count},
                        current_time=time.time(),
                    )
                except Exception:
                    pass

        # Final save
        self._manual_save()

    def _chat(self, message: str) -> None:
        """Send a message to Jack and print the response."""
        # Emotional event (update brain's emotional state)
        if self.brain.emotional_state is not None and EventType is not None:
            try:
                self.brain.emotional_state.update(
                    event_type=EventType.USER_CHAT,
                    user_interaction=0.8,
                )
            except Exception:
                pass

        # Route: task commands → TaskManager, conversation → chat
        response = ""
        msg_lower = message.lower().strip()
        try:
            is_task = any(kw in msg_lower for kw in [
                "make ", "fetch ", "bring ", "clean ", "go make", "go get",
                "pick up", "put ", "carry ", "take ", "move ",
            ])
            if is_task and self.task_manager is not None:
                response = self.task_manager.set_task(message)
            elif msg_lower in ("stop", "cancel", "nevermind"):
                response = self.task_manager.cancel() if self.task_manager else "I wasn't doing anything."
            elif hasattr(self.brain, "chat"):
                response = self.brain.chat(message, speak=False)
            else:
                response = "I heard you! Still learning to respond properly."
        except Exception as exc:
            response = f"[Error: {exc}] I'm having trouble thinking right now."

        print(f"Jack> {response}")

        # Memory - store both sides of conversation
        try:
            if hasattr(self.brain, "remember"):
                self.brain.remember(f"User said: {message}")
                if response:
                    self.brain.remember(f"I replied: {response}", importance=0.3)
        except Exception:
            pass

        # Update emotional state with decay
        if self.brain.emotional_state is not None:
            try:
                self.brain.emotional_state.update(dt=1.0)
            except Exception:
                pass

    def _show_mood(self) -> None:
        """Display current emotional state from brain."""
        if self.brain.emotional_state is None:
            print("[No emotional state module loaded]")
            return

        try:
            mood = self.brain.emotional_state.get_mood_dict()
            dominant = self.brain.emotional_state.get_dominant_mood()

            print(f"\n  Mood: {dominant}")
            print(f"  Pleasure:  {mood['pleasure']:+.3f}  {'#' * int((mood['pleasure'] + 1) * 10)}")
            print(f"  Arousal:   {mood['arousal']:+.3f}  {'#' * int((mood['arousal'] + 1) * 10)}")
            print(f"  Dominance: {mood['dominance']:+.3f}  {'#' * int((mood['dominance'] + 1) * 10)}")
            print()
        except Exception as exc:
            print(f"[Mood error: {exc}]")

    def _trigger_thought(self) -> None:
        """Force an inner monologue thought using brain's modules."""
        if self.brain.inner_monologue is None:
            print("[No inner monologue module loaded]")
            return

        try:
            mood_dict = {}
            if self.brain.emotional_state is not None:
                mood_dict = self.brain.emotional_state.get_mood_dict()

            personality_prompt = ""
            if self.brain.personality is not None:
                personality_prompt = self.brain.personality.get_inner_monologue_prompt(
                    mood_dict=mood_dict,
                )

            thought = self.brain.inner_monologue.think(
                personality_prompt=personality_prompt,
                mood_dict=mood_dict,
                recent_memories=[],
                current_goal="explore",
            )
            print(f"  [Jack thinks: \"{thought}\"]")

            # Store thought in brain's long-term memory
            if thought and hasattr(self.brain, 'memory') and self.brain.memory is not None:
                self.brain.memory.add(f"[Thought] {thought}", importance=0.5)

        except Exception as exc:
            print(f"[Thought error: {exc}]")

    def _manual_save(self) -> None:
        """Perform a manual save."""
        if self.persistence is None:
            print("[No persistence module loaded]")
            return

        try:
            path = self.persistence.save_all(
                self.brain,
                world_state={"mode": "text", "step": self.step_count},
            )
            print(f"[Saved to {path}]")
        except Exception as exc:
            print(f"[Save failed: {exc}]")

    def _show_status(self) -> None:
        """Display brain and module status."""
        print(f"\n  Brain: {type(self.brain).__name__}")
        if torch is not None:
            n_params = sum(p.numel() for p in self.brain.parameters())
            print(f"  Parameters: {n_params:,}")
        print(f"  Emotional State: {'loaded' if self.brain.emotional_state else 'none'}")
        print(f"  Personality: {'loaded' if self.brain.personality else 'none'}")
        print(f"  Inner Monologue: {'loaded' if self.brain.inner_monologue else 'none'}")
        print(f"  Persistence: {'loaded' if self.persistence else 'none'}")
        print(f"  Steps: {self.step_count}")

        if hasattr(self.brain, "memory") and self.brain.memory is not None:
            n_mem = len(getattr(self.brain.memory, "memories", []))
            print(f"  Memories: {n_mem}")
        print()


# =============================================================================
# FACTORY: Create the appropriate world based on available dependencies
# =============================================================================

def create_world(
    brain: UnifiedBrain,
    config: Optional[WorldConfig] = None,
    force_text: bool = False,
) -> Any:
    """Create a VirtualWorld (GUI) or TextOnlyWorld (fallback).

    Selects the best available mode:
    1. If force_text is True, always use TextOnlyWorld.
    2. If both PyGame and MuJoCo are available, use VirtualWorld (full GUI).
    3. Otherwise, fall back to TextOnlyWorld.

    The brain is the SINGLE source of truth for all companion modules
    (emotional state, personality, inner monologue, movement-mood coupling).
    These are created in UnifiedBrain.__init__, NOT here. The factory just
    creates the brain and passes it to the world.

    Args:
        brain: The UnifiedBrain instance (already owns all companion modules).
        config: WorldConfig for display/physics settings.
        force_text: If True, skip GUI and use text mode.

    Returns:
        Either a VirtualWorld or TextOnlyWorld instance.
    """
    config = config or WorldConfig()

    # ── Log what the brain already has ────────────────────────────────────
    print("[Factory] Brain companion modules (single source of truth):")
    print(f"  EmotionalState: {'loaded' if getattr(brain, 'emotional_state', None) else 'none'}")
    print(f"  Personality: {'loaded' if getattr(brain, 'personality', None) else 'none'}")
    print(f"  InnerMonologue: {'loaded' if getattr(brain, 'inner_monologue', None) else 'none'}")
    print(f"  MovementMoodCoupling: {'loaded' if getattr(brain, 'movement_mood', None) else 'none'}")
    print(f"  Memory: {'loaded' if getattr(brain, 'memory', None) else 'none'}")

    # ── Select world type ─────────────────────────────────────────────────
    if force_text:
        print("[Factory] Text-only mode (forced)")
        return TextOnlyWorld(
            brain=brain,
            save_dir=config.save_dir,
        )

    if HAS_PYGAME and HAS_MUJOCO:
        print("[Factory] Full GUI mode (PyGame + MuJoCo)")
        return VirtualWorld(
            brain=brain,
            config=config,
        )

    # Fallback
    missing = []
    if not HAS_PYGAME:
        missing.append("pygame")
    if not HAS_MUJOCO:
        missing.append("mujoco")
    print(f"[Factory] Text-only mode (missing: {', '.join(missing)})")

    return TextOnlyWorld(
        brain=brain,
        save_dir=config.save_dir,
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    """Command-line entry point for Jack's Virtual World.

    Usage:
        python VirtualWorld.py                           # Default scene, GUI
        python VirtualWorld.py --scene room              # Room scene
        python VirtualWorld.py --scene humanoid           # Humanoid scene
        python VirtualWorld.py --scene terrain            # Terrain scene
        python VirtualWorld.py --scene path/to/model.xml  # Custom XML
        python VirtualWorld.py --text-only               # Text mode
        python VirtualWorld.py --load saves/latest.pt    # Load a save
        python VirtualWorld.py --width 1280 --height 720 # Custom resolution
    """
    parser = argparse.ArgumentParser(
        description="Jack's Virtual World - AI Companion Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python VirtualWorld.py                    Launch with default scene
  python VirtualWorld.py --scene room       Launch room scene
  python VirtualWorld.py --text-only        Text-only interaction
  python VirtualWorld.py --load saves/x.pt  Resume from save
        """,
    )

    parser.add_argument(
        "--scene", type=str, default=None,
        help=(
            f"Scene to load. Built-in: {list(SCENE_CATALOG.keys())}. "
            "Or provide a path to a custom MuJoCo XML file."
        ),
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Use text-only mode (no GUI rendering).",
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Path to a save file (.pt) to resume from.",
    )
    parser.add_argument(
        "--width", type=int, default=800,
        help="Window width in pixels (default: 800).",
    )
    parser.add_argument(
        "--height", type=int, default=600,
        help="Window height in pixels (default: 600).",
    )
    parser.add_argument(
        "--fps", type=int, default=50,
        help="Target frames per second (default: 50).",
    )
    parser.add_argument(
        "--save-dir", type=str, default="saves",
        help="Directory for save files (default: saves/).",
    )

    args = parser.parse_args()

    # ── Configure logging ─────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Resolve scene path ────────────────────────────────────────────────
    scene_xml = None
    if args.scene is not None:
        if args.scene in SCENE_CATALOG:
            scene_xml = SCENE_CATALOG[args.scene]
            print(f"[Main] Scene: {args.scene} -> {scene_xml}")
        elif os.path.isfile(args.scene):
            scene_xml = os.path.abspath(args.scene)
            print(f"[Main] Custom scene: {scene_xml}")
        else:
            print(f"[Main] Warning: scene '{args.scene}' not found. Using default.")

    # ── Build configuration ───────────────────────────────────────────────
    world_config = WorldConfig(
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        scene_xml=scene_xml,
        save_dir=args.save_dir,
    )

    # ── Create brain ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  JACK'S VIRTUAL WORLD")
    print("  Creating UnifiedBrain...")
    print("=" * 70 + "\n")

    brain_config = UnifiedBrainConfig()

    # Disable heavy features for interactive use (faster startup)
    brain_config.use_pretrained_vision = False
    brain_config.use_pretrained_audio = False

    if torch is not None:
        brain = UnifiedBrain(brain_config)
        brain.eval()  # Inference mode
    else:
        print("[Main] PyTorch not available. Brain will be non-functional.")
        # Create a minimal stub -- TextOnlyWorld handles the None case
        brain = UnifiedBrain(brain_config)

    # ── Load save if requested ────────────────────────────────────────────
    if args.load is not None:
        if CompanionPersistence is not None:
            try:
                persistence = CompanionPersistence(SaveConfig(save_dir=args.save_dir))
                info = persistence.load_all(brain, path=args.load)
                print(f"[Main] Loaded save: v{info['version']}, "
                      f"{info['num_memories']} memories")
            except Exception as exc:
                print(f"[Main] Failed to load save: {exc}")
        else:
            print("[Main] Persistence module not available -- cannot load save.")

    # ── Create and run the world ──────────────────────────────────────────
    world = create_world(
        brain=brain,
        config=world_config,
        force_text=args.text_only,
    )

    print("\n" + "=" * 70)
    print(f"  Starting {'Text Mode' if args.text_only else 'Virtual World'}...")
    print("=" * 70 + "\n")

    world.run()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
