"""
Reward calculator for RL meme pipeline.

Computes rewards by comparing facial reactions to baseline emotional state.
"""

import time
from typing import Optional, List
from dataclasses import dataclass, asdict

from loguru import logger

from .emotion_analyzer import EmotionResult, BaselineResult, EmotionWeights


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    scale_factor: float = 2.0  # Multiplier for delta
    clip_min: float = -1.0
    clip_max: float = 1.0
    min_face_ratio: float = 0.3  # Minimum frames with face detected
    no_face_penalty: float = 0.0  # Reward when no face detected
    decay_factor: float = 0.95  # For temporal weighting (recent frames matter more)
    emotion_weights: EmotionWeights = None

    def __post_init__(self):
        if self.emotion_weights is None:
            self.emotion_weights = EmotionWeights()


@dataclass
class RewardResult:
    """Computed reward from comparing reaction to baseline."""
    reward: float  # Clipped to [-1, 1]
    raw_delta: float  # Unclipped reaction_score - baseline
    reaction_score: float
    baseline_score: float
    timestamp: float
    face_ratio: float  # Ratio of frames with face detected
    meme_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class RewardCalculator:
    """
    Calculates RL rewards from facial reactions.

    Formula: reward = clip((reaction_score - baseline) * scale_factor, -1, 1)

    Features:
    - Baseline comparison
    - Configurable clipping and scaling
    - Temporal weighting (recent reactions weighted more)
    - Handles missing face detection gracefully
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward calculator.

        Args:
            config: Reward calculation configuration
        """
        self.config = config or RewardConfig()
        self._baseline: Optional[BaselineResult] = None
        self._history: List[RewardResult] = []

    def set_baseline(self, baseline: BaselineResult) -> None:
        """
        Set the baseline emotional state for comparison.

        Args:
            baseline: BaselineResult from EmotionAnalyzer.capture_baseline()
        """
        if not baseline.valid:
            logger.warning("Setting invalid baseline - rewards may be inaccurate")

        self._baseline = baseline
        logger.info(f"Baseline set: amusement={baseline.avg_amusement:.3f}")

    def clear_baseline(self) -> None:
        """Clear the current baseline."""
        self._baseline = None

    @property
    def has_baseline(self) -> bool:
        """Check if a valid baseline is set."""
        return self._baseline is not None and self._baseline.valid

    def compute_reward(
        self,
        results: List[EmotionResult],
        meme_id: Optional[str] = None
    ) -> RewardResult:
        """
        Compute reward from emotion analysis results.

        Args:
            results: List of EmotionResult from reaction capture
            meme_id: Optional identifier for the meme shown

        Returns:
            RewardResult with computed reward

        Raises:
            ValueError: If no baseline is set
        """
        if self._baseline is None:
            raise ValueError("No baseline set - call set_baseline() first")

        timestamp = time.time()

        # Check face detection ratio
        valid_results = [r for r in results if r.face_detected]
        face_ratio = len(valid_results) / len(results) if results else 0.0

        if face_ratio < self.config.min_face_ratio:
            logger.warning(f"Low face detection ratio: {face_ratio:.2f}")
            reward_result = RewardResult(
                reward=self.config.no_face_penalty,
                raw_delta=0.0,
                reaction_score=0.0,
                baseline_score=self._baseline.avg_amusement,
                timestamp=timestamp,
                face_ratio=face_ratio,
                meme_id=meme_id
            )
            self._history.append(reward_result)
            return reward_result

        # Compute reaction score with temporal weighting
        reaction_score = self._compute_reaction_score(valid_results)
        baseline_score = self._baseline.avg_amusement

        # Compute delta and apply scaling
        raw_delta = reaction_score - baseline_score
        reward = self._apply_clipping(raw_delta)

        logger.debug(
            f"Reward: {reward:.3f} (reaction={reaction_score:.3f}, "
            f"baseline={baseline_score:.3f}, delta={raw_delta:.3f})"
        )

        reward_result = RewardResult(
            reward=reward,
            raw_delta=raw_delta,
            reaction_score=reaction_score,
            baseline_score=baseline_score,
            timestamp=timestamp,
            face_ratio=face_ratio,
            meme_id=meme_id
        )

        self._history.append(reward_result)
        return reward_result

    def _compute_reaction_score(self, results: List[EmotionResult]) -> float:
        """
        Compute aggregate reaction score from multiple frames.

        Uses temporal weighting where recent frames have higher weight.
        """
        if not results:
            return 0.0

        weights = self.config.emotion_weights

        # Apply temporal decay (most recent frames weighted more)
        scores = []
        temporal_weights = []

        for i, result in enumerate(results):
            score = result.amusement_score(weights)
            scores.append(score)
            # Exponential decay: most recent = 1.0, older frames decay
            temporal_weight = self.config.decay_factor ** (len(results) - 1 - i)
            temporal_weights.append(temporal_weight)

        # Weighted average
        total_weight = sum(temporal_weights)
        if total_weight == 0:
            return sum(scores) / len(scores)

        weighted_sum = sum(s * w for s, w in zip(scores, temporal_weights))
        return weighted_sum / total_weight

    def _apply_clipping(self, raw_delta: float) -> float:
        """Apply scaling and clipping to raw delta."""
        scaled = raw_delta * self.config.scale_factor
        return max(self.config.clip_min, min(self.config.clip_max, scaled))

    def get_history(self, n: Optional[int] = None) -> List[RewardResult]:
        """
        Get recent reward history.

        Args:
            n: Number of recent rewards, None for all

        Returns:
            List of RewardResult objects
        """
        if n is None:
            return list(self._history)
        return list(self._history[-n:])

    def get_average_reward(self, n: Optional[int] = None) -> float:
        """Get average reward from history."""
        history = self.get_history(n)
        if not history:
            return 0.0
        return sum(r.reward for r in history) / len(history)

    def clear_history(self) -> None:
        """Clear reward history."""
        self._history.clear()
