"""
PERSISTENCE SYSTEM - Unified Save/Load for Virtual Companion

Research backing:
- Park et al. (2023): Generative Agents - persistent memory across sessions
  https://arxiv.org/abs/2304.03442
- MemGPT (Packer et al. 2023): Persistent agent state management
  https://arxiv.org/abs/2310.08560

Architecture:
- Single save file contains ALL companion state
- Atomic writes prevent corruption on crash
- Version field enables forward-compatible loading
- Auto-save runs periodically during game loop

Save format (dict saved as .pt):
{
    'version': '1.0',
    'timestamp': float,
    'model_weights': state_dict,
    'obs_projection_weights': state_dict,
    'emotional_state': {pad_vector, history, baseline},
    'personality': {traits, speech_style, backstory},
    'memories': [{text, embedding, timestamp, importance}],
    'inner_monologue_history': [(timestamp, text, type)],
    'world_state': {jack_position, objects, time_of_day},
    'training_state': {phase, epoch, global_step},
}

Author: Janno Louwrens
"""

import os
import time
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# SAVE FORMAT VERSION HISTORY
# ==============================================================================
# 1.0 - Initial: model weights, memories, world state, training state
# 1.1 - Added: emotional_state, personality, inner_monologue_history
# 1.2 - Added: obs_projection_weights, autonomous_mind state
# ==============================================================================

CURRENT_VERSION = "1.2"

COMPATIBLE_VERSIONS = {"1.0", "1.1", "1.2"}


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class SaveConfig:
    """
    Configuration for the persistence system.

    Attributes:
        save_dir: Directory where save files are stored.
        auto_save_interval: Seconds between automatic saves (default 5 minutes).
        max_saves: Maximum number of save files to keep. Oldest are pruned when
                   this limit is exceeded.
        save_prefix: Filename prefix for save files.
        compress: Whether to use torch.save compression (reduces file size but
                  increases save/load time slightly).
    """
    save_dir: str = "saves"
    auto_save_interval: float = 300.0  # 5 minutes
    max_saves: int = 10
    save_prefix: str = "jack_companion"
    compress: bool = False


# ==============================================================================
# PERSISTENCE SYSTEM
# ==============================================================================

class CompanionPersistence:
    """
    Unified save/load system for Jack's complete companion state.

    Handles serialisation of the entire UnifiedBrain state including:
    - Neural network weights (model_state_dict)
    - Companion memory (long-term facts and embeddings)
    - World state (positions, objects, time)
    - Training progress (phase, epoch, global step)
    - Emotional state (PAD vector, history, baseline)
    - Personality (traits, speech style, backstory)
    - Inner monologue history

    All writes are atomic: data is written to a temporary file first, then
    renamed to the target path. This prevents corruption if the process is
    killed mid-save.

    Inspired by:
    - Park et al. (2023) Generative Agents: persistent memory retrieval
    - Packer et al. (2023) MemGPT: managing long-term agent state
    """

    def __init__(self, config: SaveConfig = None):
        self.config = config or SaveConfig()
        self._last_auto_save: float = 0.0
        self._save_count: int = 0

        # Ensure save directory exists
        os.makedirs(self.config.save_dir, exist_ok=True)
        logger.info(
            "Persistence initialised: dir=%s, auto_save=%ds, max_saves=%d",
            self.config.save_dir,
            int(self.config.auto_save_interval),
            self.config.max_saves,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def save_all(
        self,
        brain: "UnifiedBrain",
        world_state: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> str:
        """
        Save the complete companion state to disk.

        Args:
            brain: The UnifiedBrain instance whose state is being saved.
            world_state: Optional dictionary describing the current world
                         (jack_position, objects, time_of_day, etc.).
            path: Explicit file path. If None a timestamped name is generated
                  inside ``config.save_dir``.

        Returns:
            The absolute path of the saved file.

        Raises:
            RuntimeError: If the atomic write fails (e.g. disk full).
        """
        if path is None:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.save_prefix}_{timestamp_str}.pt"
            path = os.path.join(self.config.save_dir, filename)

        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = self._collect_state(brain, world_state)

        self._atomic_save(save_data, path)

        self._save_count += 1
        self._last_auto_save = time.time()

        # Prune old saves if over limit
        self._prune_old_saves()

        logger.info("[SAVE] Complete companion state -> %s", path)
        return path

    def load_all(
        self,
        brain: "UnifiedBrain",
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a complete companion state and apply it to the brain.

        Args:
            brain: The UnifiedBrain instance to restore state into.
            path: Explicit file path. If None the latest save is loaded.

        Returns:
            A dictionary with metadata about the loaded state:
            ``{'version', 'timestamp', 'training_state', 'num_memories',
               'world_state_present'}``.

        Raises:
            FileNotFoundError: If no save file is found.
            RuntimeError: If the save file is corrupt or incompatible.
        """
        if path is None:
            path = self.get_latest_save()
            if path is None:
                raise FileNotFoundError(
                    f"No save files found in {self.config.save_dir}"
                )

        path = os.path.abspath(path)
        logger.info("[LOAD] Loading companion state from %s", path)

        # Load raw data
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load save file {path}: {exc}"
            ) from exc

        # Validate
        if not isinstance(data, dict) or "version" not in data:
            raise RuntimeError(
                f"Invalid save file format: {path} (missing version field)"
            )

        version = data["version"]
        if version not in COMPATIBLE_VERSIONS:
            raise RuntimeError(
                f"Incompatible save version {version}. "
                f"Supported: {COMPATIBLE_VERSIONS}"
            )

        # Migrate if needed
        if version != CURRENT_VERSION:
            logger.info(
                "Migrating save from v%s -> v%s", version, CURRENT_VERSION
            )
            data = self._migrate_save(data, version, CURRENT_VERSION)

        # Apply state
        self._apply_state(brain, data)

        info = {
            "version": data.get("version", "unknown"),
            "timestamp": data.get("timestamp", 0.0),
            "training_state": data.get("training_state", {}),
            "num_memories": len(data.get("memories", [])),
            "world_state_present": "world_state" in data
                                    and data["world_state"] is not None,
        }

        logger.info(
            "[LOAD] Restored: v%s, %d memories, training phase %s",
            info["version"],
            info["num_memories"],
            info["training_state"].get("phase", "unknown"),
        )
        return info

    def auto_save_tick(
        self,
        brain: "UnifiedBrain",
        world_state: Optional[Dict[str, Any]],
        current_time: float,
    ) -> bool:
        """
        Check whether enough time has elapsed and perform an auto-save.

        Call this once per game-loop tick. It is cheap when no save is needed
        (single float comparison).

        Args:
            brain: The UnifiedBrain instance.
            world_state: Current world state dictionary.
            current_time: Monotonic time (``time.time()`` or game clock).

        Returns:
            True if a save was performed, False otherwise.
        """
        elapsed = current_time - self._last_auto_save
        if elapsed < self.config.auto_save_interval:
            return False

        try:
            auto_path = os.path.join(
                self.config.save_dir,
                f"{self.config.save_prefix}_autosave.pt",
            )
            self.save_all(brain, world_state, path=auto_path)
            logger.info("[AUTO-SAVE] Saved after %.0fs elapsed", elapsed)
            return True
        except Exception:
            logger.exception("[AUTO-SAVE] Failed")
            return False

    def list_saves(self) -> List[Dict[str, Any]]:
        """
        List all available save files sorted by timestamp (newest first).

        Returns:
            List of dicts with keys: ``path``, ``timestamp``, ``version``,
            ``size_bytes``, ``filename``.
        """
        saves = []
        save_dir = self.config.save_dir

        if not os.path.isdir(save_dir):
            return saves

        for filename in os.listdir(save_dir):
            if not filename.endswith(".pt"):
                continue

            filepath = os.path.join(save_dir, filename)
            if not os.path.isfile(filepath):
                continue

            info = self._peek_save(filepath)
            if info is not None:
                saves.append(info)

        # Sort newest first
        saves.sort(key=lambda s: s["timestamp"], reverse=True)
        return saves

    def get_latest_save(self) -> Optional[str]:
        """
        Return the path to the most recent save file, or None.
        """
        saves = self.list_saves()
        if not saves:
            return None
        return saves[0]["path"]

    # ------------------------------------------------------------------
    # STATE COLLECTION (brain -> dict)
    # ------------------------------------------------------------------

    def _collect_state(
        self,
        brain: "UnifiedBrain",
        world_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Gather all saveable state from the brain into a single dictionary.
        """
        data: Dict[str, Any] = {
            "version": CURRENT_VERSION,
            "timestamp": time.time(),
        }

        # --- Model weights ---
        try:
            data["model_weights"] = brain.state_dict()
        except Exception:
            logger.warning("Could not serialise model weights")
            data["model_weights"] = None

        # --- Observation projection weights ---
        # Some configurations have a separate obs_projection module
        if hasattr(brain, "obs_projection"):
            try:
                data["obs_projection_weights"] = (
                    brain.obs_projection.state_dict()
                )
            except Exception:
                data["obs_projection_weights"] = None
        else:
            data["obs_projection_weights"] = None

        # --- Emotional state ---
        data["emotional_state"] = self._collect_emotional_state(brain)

        # --- Personality ---
        data["personality"] = self._collect_personality(brain)

        # --- Memories (CompanionMemory) ---
        data["memories"] = self._collect_memories(brain)

        # --- Inner monologue history ---
        data["inner_monologue_history"] = self._collect_inner_monologue(brain)

        # --- World state ---
        data["world_state"] = world_state

        # --- Training state ---
        data["training_state"] = self._collect_training_state(brain)

        # --- Autonomous mind / intrinsic motivation ---
        data["autonomous_mind_state"] = self._collect_autonomous_mind(brain)

        return data

    def _collect_emotional_state(
        self, brain: "UnifiedBrain"
    ) -> Optional[Dict[str, Any]]:
        """Extract emotional state (PAD model) if present."""
        if not hasattr(brain, "emotional_state"):
            return None

        emo = brain.emotional_state
        if emo is None:
            return None

        try:
            state: Dict[str, Any] = {}
            # PAD vector (Pleasure-Arousal-Dominance)
            if hasattr(emo, "pad_vector"):
                pad = emo.pad_vector
                state["pad_vector"] = (
                    pad.cpu().tolist() if torch.is_tensor(pad) else list(pad)
                )
            # History of emotional states. EmotionalState.history is a
            # MoodHistory (not iterable itself — its entries deque is);
            # list(MoodHistory) raises TypeError, which the except below
            # turned into silently dropping the WHOLE emotional state
            # from every save (caught by T6.03).
            if hasattr(emo, "history"):
                hist = emo.history
                state["history"] = list(getattr(hist, "entries", hist))
            # Baseline / resting state
            if hasattr(emo, "baseline"):
                bl = emo.baseline
                state["baseline"] = (
                    bl.cpu().tolist() if torch.is_tensor(bl) else list(bl)
                )
            # GRU hidden state (temporal dynamics)
            if hasattr(emo, '_gru_hidden'):
                state['gru_hidden'] = emo._gru_hidden.cpu().tolist()
            return state if state else None
        except Exception:
            logger.warning("Could not serialise emotional state")
            return None

    def _collect_personality(
        self, brain: "UnifiedBrain"
    ) -> Optional[Dict[str, Any]]:
        """Extract personality traits if present."""
        if not hasattr(brain, "personality"):
            return None

        pers = brain.personality
        if pers is None:
            return None

        try:
            state: Dict[str, Any] = {}
            # Personality stores everything in pers.config (PersonalityConfig)
            if hasattr(pers, "config") and pers.config is not None:
                cfg = pers.config
                state["traits"] = {
                    "openness": cfg.openness,
                    "conscientiousness": cfg.conscientiousness,
                    "extraversion": cfg.extraversion,
                    "agreeableness": cfg.agreeableness,
                    "neuroticism": cfg.neuroticism,
                }
                state["name"] = cfg.name
                state["backstory"] = cfg.backstory
            return state if state else None
        except Exception:
            logger.warning("Could not serialise personality")
            return None

    def _collect_memories(
        self, brain: "UnifiedBrain"
    ) -> List[Dict[str, Any]]:
        """
        Extract memories from CompanionMemory.

        Each memory is stored as:
        {text: str, embedding: list|None, timestamp: float, importance: float}
        """
        if not hasattr(brain, "memory") or brain.memory is None:
            return []

        memories = []
        try:
            raw_memories = getattr(brain.memory, "memories", [])
            for mem in raw_memories:
                serialised = {
                    "text": mem.get("text", ""),
                    "timestamp": mem.get("timestamp", 0.0),
                    "importance": mem.get("importance", 1.0),
                }
                # Embeddings: convert numpy arrays to lists for serialisation
                emb = mem.get("embedding")
                if emb is not None:
                    if isinstance(emb, np.ndarray):
                        serialised["embedding"] = emb.tolist()
                    elif torch.is_tensor(emb):
                        serialised["embedding"] = emb.cpu().tolist()
                    else:
                        serialised["embedding"] = emb
                else:
                    serialised["embedding"] = None
                memories.append(serialised)
        except Exception:
            logger.warning("Could not serialise memories")

        return memories

    def _collect_inner_monologue(
        self, brain: "UnifiedBrain"
    ) -> List[Tuple[float, str, str]]:
        """Extract inner monologue history if present."""
        # The actual attribute path is brain.inner_monologue.thought_history
        if not hasattr(brain, "inner_monologue") or brain.inner_monologue is None:
            return []

        try:
            history = brain.inner_monologue.thought_history
            if history is None:
                return []
            # Each entry: (timestamp, text, type)
            return [(float(ts), str(txt), str(tp)) for ts, txt, tp in history]
        except Exception:
            logger.warning("Could not serialise inner monologue history")
            return []

    def _collect_training_state(
        self, brain: "UnifiedBrain"
    ) -> Dict[str, Any]:
        """Extract training-related metadata."""
        state: Dict[str, Any] = {}

        # These may be set by RobustTrainer or externally
        for attr in ("current_phase", "training_phase", "phase"):
            if hasattr(brain, attr):
                state["phase"] = getattr(brain, attr)
                break

        for attr in ("epoch", "current_epoch"):
            if hasattr(brain, attr):
                state["epoch"] = getattr(brain, attr)
                break

        if hasattr(brain, "global_step"):
            state["global_step"] = brain.global_step

        return state

    def _collect_autonomous_mind(
        self, brain: "UnifiedBrain"
    ) -> Optional[Dict[str, Any]]:
        """Extract AutonomousMind / intrinsic motivation state."""
        if not hasattr(brain, "autonomous_mind") or brain.autonomous_mind is None:
            return None

        mind = brain.autonomous_mind
        try:
            state: Dict[str, Any] = {}
            if hasattr(mind, "curiosity_scores"):
                state["curiosity_scores"] = list(mind.curiosity_scores)
            if hasattr(mind, "skill_library"):
                state["skill_library"] = list(mind.skill_library)
            if hasattr(mind, "metacognition_log"):
                state["metacognition_log"] = list(mind.metacognition_log)
            return state if state else None
        except Exception:
            logger.warning("Could not serialise autonomous mind state")
            return None

    # ------------------------------------------------------------------
    # STATE APPLICATION (dict -> brain)
    # ------------------------------------------------------------------

    def _apply_state(
        self,
        brain: "UnifiedBrain",
        data: Dict[str, Any],
    ) -> None:
        """
        Apply loaded state back onto the brain.
        Gracefully skips any component that is missing or fails.
        """
        # --- Model weights ---
        weights = data.get("model_weights")
        if weights is not None:
            try:
                brain.load_state_dict(weights, strict=False)
                logger.info("  Restored model weights (strict=False)")
            except Exception as exc:
                logger.warning("  Could not restore model weights: %s", exc)

        # --- Observation projection ---
        obs_weights = data.get("obs_projection_weights")
        if obs_weights is not None and hasattr(brain, "obs_projection"):
            try:
                brain.obs_projection.load_state_dict(obs_weights, strict=False)
                logger.info("  Restored obs_projection weights")
            except Exception as exc:
                logger.warning(
                    "  Could not restore obs_projection: %s", exc
                )

        # --- Emotional state ---
        emo_data = data.get("emotional_state")
        if emo_data is not None and hasattr(brain, "emotional_state"):
            self._apply_emotional_state(brain, emo_data)

        # --- Personality ---
        pers_data = data.get("personality")
        if pers_data is not None and hasattr(brain, "personality"):
            self._apply_personality(brain, pers_data)

        # --- Memories ---
        mem_data = data.get("memories", [])
        if mem_data and hasattr(brain, "memory") and brain.memory is not None:
            self._apply_memories(brain, mem_data)

        # --- Inner monologue ---
        mono_data = data.get("inner_monologue_history", [])
        if mono_data and hasattr(brain, "inner_monologue") and brain.inner_monologue is not None:
            try:
                from collections import deque
                brain.inner_monologue.thought_history = deque(
                    [(ts, txt, tp) for ts, txt, tp in mono_data],
                    maxlen=getattr(brain.inner_monologue.thought_history, 'maxlen', 1000),
                )
                logger.info(
                    "  Restored %d inner monologue entries", len(mono_data)
                )
            except Exception as exc:
                logger.warning(
                    "  Could not restore inner monologue: %s", exc
                )

        # --- Training state ---
        ts_data = data.get("training_state", {})
        if ts_data:
            self._apply_training_state(brain, ts_data)

        # --- Autonomous mind ---
        mind_data = data.get("autonomous_mind_state")
        if mind_data is not None:
            self._apply_autonomous_mind(brain, mind_data)

    def _apply_emotional_state(
        self, brain: "UnifiedBrain", emo_data: Dict[str, Any]
    ) -> None:
        """Restore emotional state onto the brain."""
        emo = brain.emotional_state
        if emo is None:
            return

        try:
            if "pad_vector" in emo_data and hasattr(emo, "pad_vector"):
                pad = emo_data["pad_vector"]
                if torch.is_tensor(emo.pad_vector):
                    emo.pad_vector = torch.tensor(
                        pad, dtype=emo.pad_vector.dtype
                    )
                else:
                    emo.pad_vector = pad

            if "history" in emo_data and hasattr(emo, "history"):
                hist = emo.history
                if hasattr(hist, "entries"):
                    # MoodHistory: refill in place so .record()/.get_recent()
                    # keep working after a restore (replacing it with a plain
                    # list would break the API).
                    hist.entries.clear()
                    hist.entries.extend(emo_data["history"])
                    if emo_data["history"]:
                        last = emo_data["history"][-1]
                        if isinstance(last, dict) and "timestamp" in last:
                            hist._last_record_time = last["timestamp"]
                else:
                    emo.history = list(emo_data["history"])

            if "baseline" in emo_data and hasattr(emo, "baseline"):
                bl = emo_data["baseline"]
                if torch.is_tensor(emo.baseline):
                    emo.baseline = torch.tensor(bl, dtype=emo.baseline.dtype)
                else:
                    emo.baseline = bl

            # Restore GRU hidden state (temporal dynamics)
            if 'gru_hidden' in emo_data and hasattr(emo, '_gru_hidden'):
                emo._gru_hidden.copy_(torch.tensor(emo_data['gru_hidden']))

            logger.info("  Restored emotional state")
        except Exception as exc:
            logger.warning("  Could not restore emotional state: %s", exc)

    def _apply_personality(
        self, brain: "UnifiedBrain", pers_data: Dict[str, Any]
    ) -> None:
        """Restore personality onto the brain."""
        pers = brain.personality
        if pers is None:
            return

        try:
            # Personality stores everything in pers.config (PersonalityConfig)
            if hasattr(pers, "config") and pers.config is not None:
                cfg = pers.config
                if "traits" in pers_data:
                    traits = pers_data["traits"]
                    for trait_name in ("openness", "conscientiousness",
                                       "extraversion", "agreeableness",
                                       "neuroticism"):
                        if trait_name in traits:
                            setattr(cfg, trait_name, traits[trait_name])
                if "name" in pers_data:
                    cfg.name = pers_data["name"]
                if "backstory" in pers_data:
                    cfg.backstory = pers_data["backstory"]
                # Recompute PAD baseline from restored traits
                from Personality import big_five_to_pad_baseline
                pers.pad_baseline = big_five_to_pad_baseline(cfg)
            logger.info("  Restored personality")
        except Exception as exc:
            logger.warning("  Could not restore personality: %s", exc)

    def _apply_memories(
        self, brain: "UnifiedBrain", mem_data: List[Dict[str, Any]]
    ) -> None:
        """Restore CompanionMemory entries."""
        try:
            restored = []
            for mem in mem_data:
                entry = {
                    "text": mem.get("text", ""),
                    "timestamp": mem.get("timestamp", 0.0),
                    "importance": mem.get("importance", 1.0),
                }
                emb = mem.get("embedding")
                if emb is not None:
                    entry["embedding"] = np.array(emb, dtype=np.float32)
                else:
                    entry["embedding"] = None
                restored.append(entry)

            brain.memory.memories = restored
            logger.info("  Restored %d memories", len(restored))
        except Exception as exc:
            logger.warning("  Could not restore memories: %s", exc)

    def _apply_training_state(
        self, brain: "UnifiedBrain", ts_data: Dict[str, Any]
    ) -> None:
        """Restore training progress metadata."""
        try:
            phase = ts_data.get("phase")
            if phase is not None:
                for attr in ("current_phase", "training_phase", "phase"):
                    if hasattr(brain, attr):
                        setattr(brain, attr, phase)
                        break

            epoch = ts_data.get("epoch")
            if epoch is not None:
                for attr in ("epoch", "current_epoch"):
                    if hasattr(brain, attr):
                        setattr(brain, attr, epoch)
                        break

            gs = ts_data.get("global_step")
            if gs is not None and hasattr(brain, "global_step"):
                brain.global_step = gs

            logger.info(
                "  Restored training state: phase=%s, epoch=%s, step=%s",
                phase, epoch, gs,
            )
        except Exception as exc:
            logger.warning("  Could not restore training state: %s", exc)

    def _apply_autonomous_mind(
        self, brain: "UnifiedBrain", mind_data: Dict[str, Any]
    ) -> None:
        """Restore AutonomousMind state."""
        if not hasattr(brain, "autonomous_mind") or brain.autonomous_mind is None:
            return

        mind = brain.autonomous_mind
        try:
            if "curiosity_scores" in mind_data and hasattr(mind, "curiosity_scores"):
                mind.curiosity_scores = list(mind_data["curiosity_scores"])
            if "skill_library" in mind_data and hasattr(mind, "skill_library"):
                mind.skill_library = list(mind_data["skill_library"])
            if "metacognition_log" in mind_data and hasattr(mind, "metacognition_log"):
                mind.metacognition_log = list(mind_data["metacognition_log"])
            logger.info("  Restored autonomous mind state")
        except Exception as exc:
            logger.warning("  Could not restore autonomous mind: %s", exc)

    # ------------------------------------------------------------------
    # ATOMIC SAVE
    # ------------------------------------------------------------------

    def _atomic_save(self, data: Dict[str, Any], path: str) -> None:
        """
        Write save data atomically: write to a temporary file in the same
        directory, then rename. This prevents corruption if the process dies
        mid-write.

        On Windows ``os.replace`` is used which is atomic on NTFS.
        On POSIX ``os.rename`` within the same filesystem is atomic.
        """
        target_dir = os.path.dirname(path) or "."
        tmp_fd = None
        tmp_path = None

        try:
            # Create temp file in the same directory so rename is same-fs
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix=".save_", dir=target_dir
            )
            os.close(tmp_fd)
            tmp_fd = None

            # Write data
            torch.save(data, tmp_path)

            # Atomic rename
            os.replace(tmp_path, path)
            tmp_path = None  # Rename succeeded, no cleanup needed

        except Exception as exc:
            # Clean up temp file on failure
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(
                f"Atomic save failed for {path}: {exc}"
            ) from exc

        finally:
            # Ensure fd is closed even on unexpected errors
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # VERSION MIGRATION
    # ------------------------------------------------------------------

    def _migrate_save(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """
        Migrate save data forward through version chain.

        Each step adds default values for newly introduced fields so the
        rest of the loading code can proceed uniformly.
        """
        version = from_version

        # 1.0 -> 1.1: add emotional_state, personality, inner_monologue_history
        if version == "1.0":
            if "emotional_state" not in data:
                data["emotional_state"] = None
            if "personality" not in data:
                data["personality"] = None
            if "inner_monologue_history" not in data:
                data["inner_monologue_history"] = []
            version = "1.1"
            logger.info("  Migrated 1.0 -> 1.1")

        # 1.1 -> 1.2: add obs_projection_weights, autonomous_mind_state
        if version == "1.1":
            if "obs_projection_weights" not in data:
                data["obs_projection_weights"] = None
            if "autonomous_mind_state" not in data:
                data["autonomous_mind_state"] = None
            version = "1.2"
            logger.info("  Migrated 1.1 -> 1.2")

        data["version"] = version
        return data

    # ------------------------------------------------------------------
    # SAVE FILE MANAGEMENT
    # ------------------------------------------------------------------

    def _peek_save(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Read only metadata from a save file without loading full weights.

        Returns a summary dict or None if the file is unreadable.
        """
        try:
            # Use map_location='meta' trick: load to meta device to avoid
            # allocating GPU memory. Fall back to cpu if that fails.
            try:
                data = torch.load(
                    filepath, map_location="cpu", weights_only=False
                )
            except Exception:
                return None

            if not isinstance(data, dict) or "version" not in data:
                return None

            return {
                "path": os.path.abspath(filepath),
                "filename": os.path.basename(filepath),
                "version": data.get("version", "unknown"),
                "timestamp": data.get("timestamp", 0.0),
                "size_bytes": os.path.getsize(filepath),
            }
        except Exception:
            return None

    def _prune_old_saves(self) -> None:
        """
        Remove oldest save files if we exceed max_saves.

        The auto-save file is never pruned (it is overwritten in place).
        """
        saves = self.list_saves()

        # Exclude auto-save from pruning candidates
        auto_name = f"{self.config.save_prefix}_autosave.pt"
        prunable = [s for s in saves if s["filename"] != auto_name]

        if len(prunable) <= self.config.max_saves:
            return

        # Prune oldest (prunable is sorted newest-first)
        to_remove = prunable[self.config.max_saves:]
        for save_info in to_remove:
            try:
                os.unlink(save_info["path"])
                logger.info("  Pruned old save: %s", save_info["filename"])
            except OSError as exc:
                logger.warning(
                    "  Could not prune %s: %s", save_info["filename"], exc
                )


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_save(
    brain: "UnifiedBrain",
    path: str = "saves/quicksave.pt",
    world_state: Optional[Dict[str, Any]] = None,
) -> str:
    """One-liner save for scripts and notebooks."""
    persistence = CompanionPersistence(SaveConfig())
    return persistence.save_all(brain, world_state=world_state, path=path)


def quick_load(
    brain: "UnifiedBrain",
    path: str = "saves/quicksave.pt",
) -> Dict[str, Any]:
    """One-liner load for scripts and notebooks."""
    persistence = CompanionPersistence(SaveConfig())
    return persistence.load_all(brain, path=path)


# ==============================================================================
# CLI / TESTING
# ==============================================================================

if __name__ == "__main__":
    """
    Self-test: verifies round-trip save/load without requiring GPU or
    the full UnifiedBrain (uses a minimal mock).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Persistence system test")
    parser.add_argument(
        "--save-dir", default="saves/test", help="Directory for test saves"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # --- Minimal mock brain for testing ---
    class MockMemory:
        def __init__(self):
            self.memories = [
                {
                    "text": "User's name is Janno",
                    "embedding": np.random.randn(64).astype(np.float32),
                    "timestamp": time.time() - 3600,
                    "importance": 0.9,
                },
                {
                    "text": "Janno likes chess and cycling",
                    "embedding": np.random.randn(64).astype(np.float32),
                    "timestamp": time.time() - 1800,
                    "importance": 0.8,
                },
            ]

    class MockBrain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            self.memory = MockMemory()
            self.inner_monologue_history = [
                (time.time() - 60, "I wonder what Janno wants", "reflection"),
                (time.time() - 30, "I should suggest a walk", "planning"),
            ]
            self.current_phase = 3
            self.epoch = 42
            self.global_step = 12345

    # --- Run test ---
    print("=" * 60)
    print("PERSISTENCE SYSTEM - Self Test")
    print("=" * 60)

    config = SaveConfig(save_dir=args.save_dir, max_saves=3)
    persistence = CompanionPersistence(config)
    brain = MockBrain()

    world = {
        "jack_position": [1.0, 0.5, 0.0],
        "objects": ["ball", "cube"],
        "time_of_day": 14.5,
    }

    # Save
    print("\n--- SAVE ---")
    path1 = persistence.save_all(brain, world_state=world)
    print(f"Saved to: {path1}")
    assert os.path.exists(path1), "Save file not created"

    # Clear brain state
    brain.memory.memories = []
    brain.inner_monologue_history = []
    brain.current_phase = 0
    brain.epoch = 0
    brain.global_step = 0

    # Load
    print("\n--- LOAD ---")
    info = persistence.load_all(brain, path=path1)
    print(f"Loaded: {info}")

    # Verify
    assert len(brain.memory.memories) == 2, (
        f"Expected 2 memories, got {len(brain.memory.memories)}"
    )
    assert brain.memory.memories[0]["text"] == "User's name is Janno"
    assert len(brain.inner_monologue_history) == 2
    assert brain.current_phase == 3
    assert brain.epoch == 42
    assert brain.global_step == 12345
    assert info["num_memories"] == 2
    assert info["world_state_present"] is True

    # List saves
    print("\n--- LIST ---")
    saves = persistence.list_saves()
    for s in saves:
        print(f"  {s['filename']}  v{s['version']}  "
              f"{s['size_bytes']} bytes")

    # Auto-save tick
    print("\n--- AUTO-SAVE ---")
    persistence._last_auto_save = 0  # Force trigger
    did_save = persistence.auto_save_tick(brain, world, time.time())
    assert did_save, "Auto-save should have triggered"

    # Pruning test: create several saves and verify max_saves is respected
    print("\n--- PRUNING ---")
    for i in range(5):
        persistence.save_all(brain, world_state=world)
        time.sleep(0.05)  # Ensure unique timestamps

    saves_after = persistence.list_saves()
    # max_saves=3, plus autosave = 4 max
    prunable = [s for s in saves_after
                if s["filename"] != f"{config.save_prefix}_autosave.pt"]
    assert len(prunable) <= config.max_saves, (
        f"Expected <= {config.max_saves} saves, got {len(prunable)}"
    )
    print(f"  {len(saves_after)} total saves ({len(prunable)} prunable)")

    # Version migration test
    print("\n--- MIGRATION ---")
    old_data = {
        "version": "1.0",
        "timestamp": time.time(),
        "model_weights": brain.state_dict(),
        "memories": [],
        "world_state": None,
        "training_state": {"phase": 1, "epoch": 5, "global_step": 100},
    }
    migrated = persistence._migrate_save(old_data, "1.0", "1.2")
    assert migrated["version"] == "1.2"
    assert "emotional_state" in migrated
    assert "obs_projection_weights" in migrated
    assert "autonomous_mind_state" in migrated
    print("  v1.0 -> v1.2 migration: OK")

    # Cleanup
    print("\n--- CLEANUP ---")
    import shutil as _shutil
    if os.path.exists(args.save_dir):
        _shutil.rmtree(args.save_dir)
        print(f"  Removed test directory: {args.save_dir}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
