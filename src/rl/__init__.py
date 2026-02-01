"""
Reinforcement Learning module for meme generation.

Provides the Gymnasium environment and PPO agent for learning
optimal template + sound combinations based on facial reactions.
"""

from .environment import MemeEnv, EnvConfig
from .agent import MemeAgent, AgentConfig, create_agent

# Check if SB3 is available
try:
    from .agent import TrainingCallback, SB3_AVAILABLE
except ImportError:
    SB3_AVAILABLE = False
    TrainingCallback = None

__all__ = [
    # Environment
    "MemeEnv",
    "EnvConfig",
    # Agent
    "MemeAgent",
    "AgentConfig",
    "create_agent",
    "TrainingCallback",
    "SB3_AVAILABLE",
]
