"""
Pipeline module for integrating RL with meme display and facial reactions.

This module provides the main orchestrator that connects:
- RL environment (template/sound selection)
- Meme player (display + audio)
- Webcam capture (facial monitoring)
- Emotion analyzer (reaction scoring)
- Reward calculator (RL feedback)
"""

from .orchestrator import (
    MemeOrchestrator,
    OrchestratorConfig,
    SessionStats,
)

__all__ = [
    "MemeOrchestrator",
    "OrchestratorConfig",
    "SessionStats",
]
