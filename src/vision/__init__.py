"""
Vision module for facial reaction scoring.

Provides webcam capture, emotion analysis, and RL reward computation.
"""

from .webcam_capture import WebcamCapture, CaptureConfig
from .emotion_analyzer import (
    EmotionAnalyzer,
    EmotionResult,
    BaselineResult,
    AnalyzerConfig,
    EmotionWeights,
)
from .reward_calculator import RewardCalculator, RewardResult, RewardConfig

__all__ = [
    "WebcamCapture",
    "CaptureConfig",
    "EmotionAnalyzer",
    "EmotionResult",
    "BaselineResult",
    "AnalyzerConfig",
    "EmotionWeights",
    "RewardCalculator",
    "RewardResult",
    "RewardConfig",
]
