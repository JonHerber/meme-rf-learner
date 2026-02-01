"""
Emotion analyzer using DeepFace for facial expression detection.

Wraps DeepFace with caching, error handling, and batch analysis support.
"""

import time
from typing import List, Optional, Dict, TYPE_CHECKING
from dataclasses import dataclass, asdict, field

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from .webcam_capture import WebcamCapture

# Lazy import for DeepFace (heavy dependency)
_deepface = None


def _get_deepface():
    """Lazy load DeepFace module."""
    global _deepface
    if _deepface is None:
        logger.info("Loading DeepFace (this may take a moment)...")
        from deepface import DeepFace
        _deepface = DeepFace
        logger.info("DeepFace loaded")
    return _deepface


@dataclass
class EmotionWeights:
    """Configurable weights for emotion-to-amusement mapping."""
    happy: float = 0.7
    surprise: float = 0.3
    neutral: float = 0.0
    sad: float = -0.2
    angry: float = -0.3
    fear: float = -0.1
    disgust: float = -0.2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalyzerConfig:
    """Configuration for emotion analyzer."""
    detector_backend: str = "opencv"  # opencv, ssd, mtcnn, retinaface
    enforce_detection: bool = False  # Don't raise error if no face
    align: bool = True
    emotion_weights: EmotionWeights = field(default_factory=EmotionWeights)


@dataclass
class EmotionResult:
    """Result of emotion analysis on a single frame."""
    timestamp: float  # Unix timestamp
    face_detected: bool = False
    emotions: Dict[str, float] = field(default_factory=dict)
    # DeepFace returns: angry, disgust, fear, happy, sad, surprise, neutral
    dominant_emotion: Optional[str] = None
    confidence: float = 0.0
    face_region: Optional[Dict[str, int]] = None  # x, y, w, h
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def amusement_score(self, weights: Optional[EmotionWeights] = None) -> float:
        """
        Calculate composite amusement score from emotions.

        Args:
            weights: Custom weights, uses defaults if None

        Returns:
            Amusement score (can be negative for negative emotions)
        """
        if not self.face_detected:
            return 0.0

        weights = weights or EmotionWeights()

        score = 0.0
        for emotion, weight in asdict(weights).items():
            # Emotions are 0-100, normalize to 0-1
            emotion_value = self.emotions.get(emotion, 0.0) / 100.0
            score += emotion_value * weight

        return score


@dataclass
class BaselineResult:
    """Baseline emotional state captured before showing meme."""
    captured_at: float  # Unix timestamp
    frame_count: int = 0
    avg_emotions: Dict[str, float] = field(default_factory=dict)
    avg_amusement: float = 0.0
    valid: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class EmotionAnalyzer:
    """
    DeepFace wrapper for emotion analysis.

    Features:
    - Lazy loading of DeepFace model
    - Configurable emotion weights
    - Batch frame analysis with averaging
    - Graceful handling of no-face scenarios
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        Initialize the emotion analyzer.

        Args:
            config: Analyzer configuration, uses defaults if None
        """
        self.config = config or AnalyzerConfig()
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Ensure DeepFace model is loaded (first analysis warms up)."""
        if not self._model_loaded:
            # Trigger model loading with a dummy analysis
            _get_deepface()
            self._model_loaded = True

    def analyze_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> EmotionResult:
        """
        Analyze a single frame for emotions.

        Args:
            frame: BGR image as numpy array (OpenCV format)
            timestamp: Optional timestamp, uses current time if None

        Returns:
            EmotionResult with detected emotions
        """
        timestamp = timestamp or time.time()

        try:
            self._ensure_model_loaded()
            DeepFace = _get_deepface()

            # Analyze with DeepFace
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                detector_backend=self.config.detector_backend,
                enforce_detection=self.config.enforce_detection,
                align=self.config.align,
                silent=True
            )

            # DeepFace returns a list when multiple faces found
            if isinstance(results, list):
                if len(results) == 0:
                    return EmotionResult(
                        timestamp=timestamp,
                        face_detected=False
                    )
                result = results[0]  # Use first face
            else:
                result = results

            emotions = result.get('emotion', {})
            region = result.get('region', {})
            dominant = result.get('dominant_emotion', None)

            return EmotionResult(
                timestamp=timestamp,
                face_detected=True,
                emotions=emotions,
                dominant_emotion=dominant,
                confidence=emotions.get(dominant, 0.0) if dominant else 0.0,
                face_region=region if region else None
            )

        except Exception as e:
            error_msg = str(e)
            # Common case: no face detected
            if "Face could not be detected" in error_msg:
                return EmotionResult(
                    timestamp=timestamp,
                    face_detected=False
                )

            logger.warning(f"Emotion analysis failed: {error_msg}")
            return EmotionResult(
                timestamp=timestamp,
                face_detected=False,
                error=error_msg
            )

    def analyze_frames(self, frames: List[np.ndarray]) -> List[EmotionResult]:
        """
        Analyze multiple frames.

        Args:
            frames: List of BGR images

        Returns:
            List of EmotionResult objects
        """
        return [self.analyze_frame(frame) for frame in frames]

    def compute_average_emotions(self, results: List[EmotionResult]) -> Dict[str, float]:
        """
        Average emotions across multiple results (ignoring frames with no face).

        Args:
            results: List of EmotionResult objects

        Returns:
            Dictionary of averaged emotion scores
        """
        valid_results = [r for r in results if r.face_detected]

        if not valid_results:
            return {}

        all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        avg_emotions = {}

        for emotion in all_emotions:
            values = [r.emotions.get(emotion, 0.0) for r in valid_results]
            avg_emotions[emotion] = sum(values) / len(values)

        return avg_emotions

    def compute_amusement_score(
        self,
        emotions: Dict[str, float],
        weights: Optional[EmotionWeights] = None
    ) -> float:
        """
        Compute composite amusement score from emotion dictionary.

        Args:
            emotions: Dictionary of emotion name to score (0-100)
            weights: Custom weights, uses config weights if None

        Returns:
            Amusement score
        """
        weights = weights or self.config.emotion_weights

        score = 0.0
        for emotion, weight in asdict(weights).items():
            # Emotions are 0-100, normalize to 0-1
            emotion_value = emotions.get(emotion, 0.0) / 100.0
            score += emotion_value * weight

        return score

    def capture_baseline(
        self,
        webcam: 'WebcamCapture',
        duration: float = 3.0,
        sample_interval: float = 0.5
    ) -> BaselineResult:
        """
        Capture baseline emotional state over a time period.

        Args:
            webcam: WebcamCapture instance
            duration: Seconds to capture baseline
            sample_interval: Seconds between samples

        Returns:
            BaselineResult with averaged emotional state
        """
        logger.info(f"Capturing baseline for {duration}s...")

        results: List[EmotionResult] = []
        start_time = time.time()

        while time.time() - start_time < duration:
            frame_data = webcam.get_frame(timeout=1.0)

            if frame_data is not None:
                timestamp, frame = frame_data
                result = self.analyze_frame(frame, timestamp)
                results.append(result)

                if result.face_detected:
                    logger.debug(f"Baseline sample: {result.dominant_emotion}")
                else:
                    logger.debug("Baseline sample: no face detected")

            time.sleep(sample_interval)

        # Compute averages
        valid_results = [r for r in results if r.face_detected]

        if not valid_results:
            logger.warning("No faces detected during baseline capture")
            return BaselineResult(
                captured_at=start_time,
                frame_count=len(results),
                valid=False,
                error="No faces detected"
            )

        avg_emotions = self.compute_average_emotions(valid_results)
        avg_amusement = self.compute_amusement_score(avg_emotions)

        logger.info(f"Baseline captured: {len(valid_results)}/{len(results)} frames with face")
        logger.info(f"Baseline amusement: {avg_amusement:.3f}")

        return BaselineResult(
            captured_at=start_time,
            frame_count=len(results),
            avg_emotions=avg_emotions,
            avg_amusement=avg_amusement,
            valid=True
        )
