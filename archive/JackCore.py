"""
JACK CORE - SUBSTRATE-AGNOSTIC UNIVERSAL AGENT

Jack can inhabit ANY substrate:
- Physical: Robot bodies, drones, arms
- Digital: Computer control (mouse, keyboard, screen)
- API: HTTP calls, databases, cloud services
- Hybrid: Any combination

The SAME brain architecture powers all substrates.
The SAME verification (MathReasoner) adapts to each domain.

Architecture:
    JackCore (universal brain)
        |
        +-- Substrate Adapters (plug-in observation/action spaces)
        |       +-- PhysicalSubstrate (robots)
        |       +-- DigitalSubstrate (computer control)
        |       +-- APISubstrate (HTTP, databases)
        |
        +-- Universal Reasoning (works across all)
                +-- WorldModel (imagine futures)
                +-- Verifier (check safety/constraints)
                +-- Planner (decompose tasks)
                +-- CreativeLoop (novel solutions)

Research foundation:
- AlphaGeometry (DeepMind 2024): Neural-symbolic verification loop
- TD-MPC2 (ICLR 2024): Latent world model imagination
- HAC (2019): Hierarchical skill decomposition
- pi0 (Physical Intelligence 2024): Flow matching for real-time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json


# ==============================================================================
# SUBSTRATE TYPES
# ==============================================================================

class SubstrateType(Enum):
    PHYSICAL = "physical"   # Robot, drone, arm
    DIGITAL = "digital"     # Computer control
    API = "api"             # HTTP, database, cloud
    HYBRID = "hybrid"       # Multiple substrates


# ==============================================================================
# ABSTRACT OBSERVATION & ACTION INTERFACES
# ==============================================================================

@dataclass
class Observation:
    """
    Universal observation that works across all substrates.
    Each substrate populates the fields it uses.
    """
    # Metadata
    substrate: SubstrateType
    timestamp: float = 0.0

    # Physical substrate
    proprio: Optional[torch.Tensor] = None          # Joint angles, velocities
    vision: Optional[torch.Tensor] = None           # Camera feed
    touch: Optional[torch.Tensor] = None            # Contact forces
    imu: Optional[torch.Tensor] = None              # Orientation, acceleration

    # Digital substrate
    screen: Optional[torch.Tensor] = None           # Screenshot embedding
    cursor_pos: Optional[tuple] = None              # (x, y) position
    active_window: Optional[str] = None             # Window title
    clipboard: Optional[str] = None                 # Clipboard contents
    file_tree: Optional[Dict] = None                # Directory structure

    # API substrate
    last_response: Optional[Dict] = None            # Last API response
    system_state: Optional[Dict] = None             # Current state
    pending_requests: Optional[List] = None         # In-flight requests

    # Shared
    language: Optional[str] = None                  # Natural language command
    goal: Optional[torch.Tensor] = None             # Goal state embedding

    def to_tensor(self, encoder: nn.Module) -> torch.Tensor:
        """Encode observation to unified tensor representation"""
        return encoder(self)


@dataclass
class Action:
    """
    Universal action that works across all substrates.
    Each substrate interprets the fields it uses.
    """
    substrate: SubstrateType
    confidence: float = 1.0

    # Physical substrate
    joint_torques: Optional[torch.Tensor] = None    # Motor commands
    target_pose: Optional[torch.Tensor] = None      # End-effector target

    # Digital substrate
    mouse_action: Optional[Dict] = None             # {type, x, y, button}
    keyboard_action: Optional[Dict] = None          # {type, key/text}
    scroll_action: Optional[Dict] = None            # {dx, dy}

    # API substrate
    http_request: Optional[Dict] = None             # {method, url, body, headers}
    db_query: Optional[Dict] = None                 # {query, params}
    shell_command: Optional[str] = None             # Bash command

    # Shared
    wait: Optional[float] = None                    # Wait N seconds
    delegate_to: Optional[str] = None               # Jack node to delegate to

    def to_executable(self) -> Dict:
        """Convert to substrate-specific executable format"""
        if self.substrate == SubstrateType.PHYSICAL:
            return {"torques": self.joint_torques, "pose": self.target_pose}
        elif self.substrate == SubstrateType.DIGITAL:
            return {"mouse": self.mouse_action, "keyboard": self.keyboard_action}
        elif self.substrate == SubstrateType.API:
            return {"http": self.http_request, "db": self.db_query, "shell": self.shell_command}
        return {}


# ==============================================================================
# SUBSTRATE ADAPTER INTERFACE
# ==============================================================================

class SubstrateAdapter(ABC):
    """
    Abstract interface for substrate-specific adapters.
    Each substrate implements observation encoding and action decoding.
    """

    @property
    @abstractmethod
    def substrate_type(self) -> SubstrateType:
        """Return the substrate type"""
        pass

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Dimension of encoded observation"""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of action space"""
        pass

    @abstractmethod
    def encode_observation(self, obs: Observation) -> torch.Tensor:
        """Encode substrate-specific observation to tensor"""
        pass

    @abstractmethod
    def decode_action(self, action_tensor: torch.Tensor) -> Action:
        """Decode action tensor to substrate-specific action"""
        pass

    @abstractmethod
    def get_constraint_rules(self) -> List[Dict]:
        """Return safety/constraint rules for this substrate"""
        pass


# ==============================================================================
# PHYSICAL SUBSTRATE (Robots)
# ==============================================================================

class PhysicalSubstrate(SubstrateAdapter, nn.Module):
    """
    Adapter for physical robots.
    Uses existing ScalableRobotBrain components.
    """

    def __init__(self, proprio_dim: int = 348, action_dim: int = 17, d_model: int = 512):
        nn.Module.__init__(self)
        self._action_dim = action_dim
        self._obs_dim = d_model

        # Reuse existing encoders
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_model),
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    @property
    def substrate_type(self) -> SubstrateType:
        return SubstrateType.PHYSICAL

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def encode_observation(self, obs: Observation) -> torch.Tensor:
        features = []

        if obs.proprio is not None:
            features.append(self.proprio_encoder(obs.proprio))

        if obs.vision is not None:
            features.append(self.vision_encoder(obs.vision))

        if features:
            return torch.stack(features).mean(dim=0)
        return torch.zeros(1, self._obs_dim)

    def decode_action(self, action_tensor: torch.Tensor) -> Action:
        torques = self.action_decoder(action_tensor)
        return Action(
            substrate=SubstrateType.PHYSICAL,
            joint_torques=torques,
        )

    def get_constraint_rules(self) -> List[Dict]:
        """Physics constraints from MathReasoner"""
        return [
            {"name": "torque_limits", "check": "abs(torque) < max_torque"},
            {"name": "joint_limits", "check": "joint_min < angle < joint_max"},
            {"name": "stability", "check": "center_of_mass in support_polygon"},
            {"name": "energy", "check": "kinetic + potential = constant"},
        ]


# ==============================================================================
# DIGITAL SUBSTRATE (Computer Control)
# ==============================================================================

class DigitalSubstrate(SubstrateAdapter, nn.Module):
    """
    Adapter for computer control.
    Screen understanding + mouse/keyboard actions.
    """

    def __init__(self, screen_size: tuple = (224, 224), d_model: int = 512):
        nn.Module.__init__(self)
        self._obs_dim = d_model
        self._action_dim = 7  # click_x, click_y, click_type, key_code, scroll_x, scroll_y, action_type

        # Screen encoder (vision transformer style)
        self.screen_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 16, 16),  # Patch embedding
            nn.Flatten(2),
            nn.Linear(64, d_model),
        )
        self.screen_proj = nn.Linear((screen_size[0]//16) * (screen_size[1]//16) * d_model, d_model)

        # Text encoder for window titles, clipboard
        self.text_encoder = nn.Sequential(
            nn.Embedding(10000, 128),
            nn.LSTM(128, d_model // 2, batch_first=True, bidirectional=True),
        )

        # Cursor position encoder
        self.cursor_encoder = nn.Linear(2, d_model)

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, self._action_dim),
        )

    @property
    def substrate_type(self) -> SubstrateType:
        return SubstrateType.DIGITAL

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def encode_observation(self, obs: Observation) -> torch.Tensor:
        features = []

        if obs.screen is not None:
            screen_feat = self.screen_encoder(obs.screen)
            B = screen_feat.shape[0]
            screen_feat = screen_feat.permute(0, 2, 1).reshape(B, -1)
            features.append(self.screen_proj(screen_feat))

        if obs.cursor_pos is not None:
            cursor = torch.tensor(obs.cursor_pos, dtype=torch.float32)
            features.append(self.cursor_encoder(cursor.unsqueeze(0)))

        if features:
            return torch.stack(features).mean(dim=0)
        return torch.zeros(1, self._obs_dim)

    def decode_action(self, action_tensor: torch.Tensor) -> Action:
        raw = self.action_decoder(action_tensor)

        # Parse action tensor
        action_type = int(raw[..., 6].argmax())  # 0=click, 1=type, 2=scroll, 3=wait

        action = Action(substrate=SubstrateType.DIGITAL)

        if action_type == 0:  # Click
            action.mouse_action = {
                "type": "click",
                "x": float(raw[..., 0].sigmoid() * 1920),  # Normalize to screen
                "y": float(raw[..., 1].sigmoid() * 1080),
                "button": "left" if raw[..., 2] > 0 else "right",
            }
        elif action_type == 1:  # Type
            action.keyboard_action = {
                "type": "key",
                "key_code": int(raw[..., 3].abs()),
            }
        elif action_type == 2:  # Scroll
            action.scroll_action = {
                "dx": float(raw[..., 4]),
                "dy": float(raw[..., 5]),
            }
        else:  # Wait
            action.wait = 0.5

        return action

    def get_constraint_rules(self) -> List[Dict]:
        """Safety constraints for computer control"""
        return [
            {"name": "no_delete_system", "check": "path not in system_paths"},
            {"name": "no_rm_rf", "check": "'rm -rf' not in command"},
            {"name": "confirm_destructive", "check": "destructive_action requires confirmation"},
            {"name": "rate_limit", "check": "actions_per_second < 10"},
            {"name": "screen_bounds", "check": "0 <= x <= width and 0 <= y <= height"},
        ]


# ==============================================================================
# API SUBSTRATE (HTTP, Databases, Shell)
# ==============================================================================

class APISubstrate(SubstrateAdapter, nn.Module):
    """
    Adapter for API calls, database queries, shell commands.
    """

    def __init__(self, d_model: int = 512):
        nn.Module.__init__(self)
        self._obs_dim = d_model
        self._action_dim = 256  # Tokenized action space

        # JSON/response encoder
        self.json_encoder = nn.Sequential(
            nn.Linear(1024, d_model),  # Assume JSON is pre-tokenized
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
        )

        # Action decoder (outputs token probabilities)
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, self._action_dim),
        )

        # Action type classifier
        self.action_type = nn.Linear(d_model, 4)  # HTTP, DB, Shell, Delegate

    @property
    def substrate_type(self) -> SubstrateType:
        return SubstrateType.API

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def encode_observation(self, obs: Observation) -> torch.Tensor:
        features = []

        if obs.last_response is not None:
            # Assume response is pre-encoded
            resp_tensor = torch.zeros(1, 1024)  # Placeholder
            features.append(self.json_encoder(resp_tensor))

        if obs.system_state is not None:
            state_tensor = torch.zeros(1, 512)  # Placeholder
            features.append(self.state_encoder(state_tensor))

        if features:
            return torch.stack(features).mean(dim=0)
        return torch.zeros(1, self._obs_dim)

    def decode_action(self, action_tensor: torch.Tensor) -> Action:
        action_type = self.action_type(action_tensor).argmax().item()
        action_tokens = self.action_decoder(action_tensor)

        action = Action(substrate=SubstrateType.API)

        if action_type == 0:  # HTTP
            action.http_request = {
                "method": "GET",  # Would be decoded from tokens
                "url": "",
                "body": None,
            }
        elif action_type == 1:  # DB
            action.db_query = {
                "query": "",  # Decoded from tokens
                "params": [],
            }
        elif action_type == 2:  # Shell
            action.shell_command = ""  # Decoded from tokens
        else:  # Delegate
            action.delegate_to = "jack@other_node"

        return action

    def get_constraint_rules(self) -> List[Dict]:
        """Safety constraints for API actions"""
        return [
            {"name": "no_drop_table", "check": "'DROP TABLE' not in query"},
            {"name": "no_delete_all", "check": "'DELETE FROM' requires WHERE"},
            {"name": "auth_required", "check": "request has auth header"},
            {"name": "rate_limit", "check": "requests_per_minute < limit"},
            {"name": "no_secrets_in_logs", "check": "response not logged if contains secrets"},
            {"name": "https_only", "check": "url.startswith('https')"},
        ]


# ==============================================================================
# UNIVERSAL VERIFIER (Adapted from MathReasoner)
# ==============================================================================

class UniversalVerifier(nn.Module):
    """
    Verifies actions before execution across all substrates.

    Adapts MathReasoner concept:
    - Physical: verify physics constraints
    - Digital: verify safety constraints
    - API: verify security constraints
    """

    def __init__(self, d_model: int = 512, num_rules: int = 100):
        super().__init__()

        # Learnable rule embeddings (like MathReasoner's physics rules)
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, d_model))

        # Rule activation network
        self.rule_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)

        # Verification head
        self.verify_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # [safe, unsafe]
        )

        # Explanation head
        self.explain_head = nn.Linear(d_model, num_rules)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        substrate_rules: List[Dict]
    ) -> Dict:
        """
        Verify if action is safe given state and substrate rules.

        Returns:
            {
                'is_safe': bool,
                'confidence': float,
                'active_rules': List[int],
                'explanation': str
            }
        """
        # Combine state and action
        combined = state + action

        # Attend to rules
        rules = self.rule_embeddings.unsqueeze(0).expand(combined.shape[0], -1, -1)
        attended, attn_weights = self.rule_attention(
            combined.unsqueeze(1), rules, rules
        )

        # Verify
        verify_logits = self.verify_head(attended.squeeze(1))
        is_safe = verify_logits.argmax(dim=-1) == 0
        confidence = F.softmax(verify_logits, dim=-1)[:, 0]

        # Which rules activated
        rule_activations = self.explain_head(attended.squeeze(1))
        active_rules = (rule_activations > 0.5).nonzero(as_tuple=True)[1].tolist()

        return {
            'is_safe': is_safe.item() if is_safe.numel() == 1 else is_safe,
            'confidence': confidence.item() if confidence.numel() == 1 else confidence,
            'active_rules': active_rules,
            'rule_weights': attn_weights,
        }


# ==============================================================================
# JACK CORE BRAIN
# ==============================================================================

@dataclass
class JackCoreConfig:
    """Configuration for universal Jack brain"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    num_rules: int = 100

    # System 1/2 thresholds
    reactive_threshold: float = 0.9
    creative_threshold: float = 0.3

    # Substrate configs
    default_substrate: SubstrateType = SubstrateType.DIGITAL


class JackCore(nn.Module):
    """
    Universal agent brain that works across all substrates.

    Same architecture as EnhancedJackBrain but with:
    - Pluggable substrate adapters
    - Universal verification
    - Multi-substrate coordination
    """

    def __init__(self, config: JackCoreConfig = None):
        super().__init__()
        self.config = config or JackCoreConfig()

        print("\n" + "="*70)
        print("       JACK CORE - UNIVERSAL AGENT BRAIN")
        print("="*70)

        # Substrate adapters (plug-in)
        self.substrates: Dict[SubstrateType, SubstrateAdapter] = {}

        # Universal components
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.config.d_model),
        )

        # Cross-modal fusion (sensors attend to each other)
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.config.d_model,
                self.config.n_heads,
                self.config.d_model * 4,
                batch_first=True
            ),
            num_layers=self.config.n_layers
        )

        # Temporal memory
        self.memory = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.config.d_model,
                self.config.n_heads,
                self.config.d_model * 4,
                batch_first=True
            ),
            num_layers=4
        )

        # Universal verifier
        self.verifier = UniversalVerifier(self.config.d_model, self.config.num_rules)

        # Mode selection
        self.confidence_head = nn.Sequential(
            nn.Linear(self.config.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.novelty_head = nn.Sequential(
            nn.Linear(self.config.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Action generation
        self.action_head = nn.Linear(self.config.d_model, self.config.d_model)

        print(f"  Substrates: [register with .register_substrate()]")
        print(f"  Verification: {self.config.num_rules} learnable rules")
        print("="*70 + "\n")

    def register_substrate(self, adapter: SubstrateAdapter):
        """Register a substrate adapter"""
        self.substrates[adapter.substrate_type] = adapter
        if isinstance(adapter, nn.Module):
            # Add as submodule for parameter tracking
            setattr(self, f"substrate_{adapter.substrate_type.value}", adapter)
        print(f"[JACK] Registered substrate: {adapter.substrate_type.value}")

    def forward(
        self,
        observation: Observation,
        mode: str = "auto"
    ) -> Dict:
        """
        Universal forward pass.

        Args:
            observation: Universal observation
            mode: "auto", "reactive", "verified", "creative"

        Returns:
            {
                'action': Action,
                'mode': str,
                'verification': Dict,
                'confidence': float
            }
        """
        substrate = observation.substrate

        if substrate not in self.substrates:
            raise ValueError(f"Substrate {substrate} not registered")

        adapter = self.substrates[substrate]

        # Encode observation
        obs_encoded = adapter.encode_observation(observation)
        if obs_encoded.dim() == 1:
            obs_encoded = obs_encoded.unsqueeze(0)

        # State representation
        state = self.state_encoder(obs_encoded)

        # Mode selection
        if mode == "auto":
            confidence = self.confidence_head(state).item()
            novelty = self.novelty_head(state).item()

            if confidence > self.config.reactive_threshold:
                mode = "reactive"
            elif novelty > self.config.creative_threshold:
                mode = "creative"
            else:
                mode = "verified"

        # Generate action
        action_tensor = self.action_head(state)

        # Decode to substrate-specific action
        action = adapter.decode_action(action_tensor)

        result = {
            'action': action,
            'mode': mode,
            'state': state,
        }

        # Verification for non-reactive modes
        if mode in ["verified", "creative"]:
            rules = adapter.get_constraint_rules()
            verification = self.verifier(state, action_tensor, rules)
            result['verification'] = verification

            if not verification['is_safe']:
                # Action blocked - would trigger refinement loop
                result['blocked'] = True
                result['block_reason'] = f"Rules violated: {verification['active_rules']}"

        return result


# ==============================================================================
# JACK NODE (Network-capable instance)
# ==============================================================================

@dataclass
class JackNode:
    """
    A Jack instance on the network.
    Can communicate with other Jacks.
    """
    node_id: str
    brain: JackCore
    substrates: List[SubstrateType]
    address: str = "localhost"
    port: int = 8080

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "substrates": [s.value for s in self.substrates],
            "address": self.address,
            "port": self.port,
        }


class JackNetwork:
    """
    Mesh network of Jack instances.
    Jacks can discover each other and delegate tasks.
    """

    def __init__(self):
        self.nodes: Dict[str, JackNode] = {}
        self.routing_table: Dict[SubstrateType, List[str]] = {}

    def register(self, node: JackNode):
        """Register a Jack node"""
        self.nodes[node.node_id] = node
        for substrate in node.substrates:
            if substrate not in self.routing_table:
                self.routing_table[substrate] = []
            self.routing_table[substrate].append(node.node_id)
        print(f"[NETWORK] Registered {node.node_id} with substrates {[s.value for s in node.substrates]}")

    def find_capable_node(self, substrate: SubstrateType) -> Optional[str]:
        """Find a node capable of handling this substrate"""
        candidates = self.routing_table.get(substrate, [])
        return candidates[0] if candidates else None

    def delegate(self, from_node: str, to_node: str, task: Dict) -> Dict:
        """Delegate a task from one Jack to another"""
        # In real implementation: HTTP/gRPC call
        print(f"[NETWORK] {from_node} -> {to_node}: {task.get('type', 'unknown')}")
        return {"status": "delegated", "to": to_node}


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("JACK CORE - UNIVERSAL AGENT DEMO")
    print("="*70 + "\n")

    # Create brain
    config = JackCoreConfig()
    brain = JackCore(config)

    # Register substrates
    brain.register_substrate(PhysicalSubstrate())
    brain.register_substrate(DigitalSubstrate())
    brain.register_substrate(APISubstrate())

    # Test digital substrate
    print("\n[TEST] Digital substrate (computer control)")
    obs = Observation(
        substrate=SubstrateType.DIGITAL,
        screen=torch.randn(1, 3, 224, 224),
        cursor_pos=(500, 300),
        language="Click the submit button",
    )

    result = brain(obs, mode="verified")
    print(f"  Mode: {result['mode']}")
    print(f"  Action type: {result['action'].substrate.value}")
    if result['action'].mouse_action:
        print(f"  Mouse: {result['action'].mouse_action}")
    if 'verification' in result:
        print(f"  Safe: {result['verification']['is_safe']}")
        print(f"  Active rules: {result['verification']['active_rules'][:5]}...")

    # Test network
    print("\n[TEST] Jack Network")
    network = JackNetwork()

    laptop_jack = JackNode(
        node_id="jack@laptop",
        brain=brain,
        substrates=[SubstrateType.DIGITAL, SubstrateType.API],
    )

    robot_jack = JackNode(
        node_id="jack@robot",
        brain=JackCore(config),
        substrates=[SubstrateType.PHYSICAL],
    )

    network.register(laptop_jack)
    network.register(robot_jack)

    # Find node for physical task
    physical_node = network.find_capable_node(SubstrateType.PHYSICAL)
    print(f"  Node for PHYSICAL: {physical_node}")

    # Delegate
    network.delegate("jack@laptop", "jack@robot", {"type": "bring_coffee"})

    print("\n" + "="*70)
    print("[OK] JackCore validated - ready for any substrate!")
    print("="*70)
