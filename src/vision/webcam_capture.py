"""
Webcam capture module with threaded frame buffer.

Provides smooth, non-blocking frame capture for real-time emotion analysis.
"""

import time
import threading
from typing import Optional, Tuple, List
from collections import deque
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger


@dataclass
class CaptureConfig:
    """Configuration for webcam capture."""
    device_id: int = 0  # Camera device index
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 10  # Max frames in buffer
    warmup_frames: int = 5  # Frames to discard on startup
    fallback_video: Optional[str] = None  # Video file for testing (WSL2 fallback)


class WebcamCapture:
    """
    Thread-safe webcam capture with frame buffer.

    Features:
    - Background thread for continuous capture
    - Ring buffer to avoid frame buildup
    - Automatic fallback to video file (for WSL2 testing)
    - Context manager support
    """

    def __init__(self, config: Optional[CaptureConfig] = None):
        """
        Initialize webcam capture.

        Args:
            config: Capture configuration, uses defaults if None
        """
        self.config = config or CaptureConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_buffer: deque = deque(maxlen=self.config.buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._using_fallback = False

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """Create and configure VideoCapture object."""
        # Try camera first
        cap = cv2.VideoCapture(self.config.device_id)

        if not cap.isOpened():
            logger.warning(f"Camera {self.config.device_id} not available")

            # Try fallback video file
            if self.config.fallback_video:
                fallback_path = Path(self.config.fallback_video)
                if fallback_path.exists():
                    logger.info(f"Using fallback video: {fallback_path}")
                    cap = cv2.VideoCapture(str(fallback_path))
                    self._using_fallback = True

            if not cap.isOpened():
                self._last_error = "No camera or fallback video available"
                return None

        # Configure capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        return cap

    def _capture_loop(self) -> None:
        """Background thread capture loop."""
        frames_captured = 0

        while self._running:
            if self._cap is None:
                break

            ret, frame = self._cap.read()

            if not ret:
                if self._using_fallback:
                    # Loop video file
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

            # Skip warmup frames
            frames_captured += 1
            if frames_captured <= self.config.warmup_frames:
                continue

            timestamp = time.time()

            with self._lock:
                self._frame_buffer.append((timestamp, frame.copy()))

            # Small sleep to prevent CPU spinning
            time.sleep(1.0 / (self.config.fps * 2))

    def start(self) -> bool:
        """
        Start the capture thread.

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            logger.warning("Capture already running")
            return True

        self._cap = self._create_capture()
        if self._cap is None:
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Wait for first frame
        timeout = 2.0
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if len(self._frame_buffer) > 0:
                    logger.info("Webcam capture started")
                    return True
            time.sleep(0.1)

        logger.warning("Timeout waiting for first frame")
        return True  # Still running, just no frames yet

    def stop(self) -> None:
        """Stop the capture thread and release resources."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        with self._lock:
            self._frame_buffer.clear()

        logger.info("Webcam capture stopped")

    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[float, np.ndarray]]:
        """
        Get the most recent frame from buffer.

        Args:
            timeout: Max seconds to wait for a frame

        Returns:
            Tuple of (timestamp, frame) or None if no frame available
        """
        start = time.time()

        while time.time() - start < timeout:
            with self._lock:
                if len(self._frame_buffer) > 0:
                    return self._frame_buffer[-1]
            time.sleep(0.05)

        return None

    def get_frames(self, n: int = 5) -> List[Tuple[float, np.ndarray]]:
        """
        Get multiple recent frames for averaging.

        Args:
            n: Number of frames to retrieve

        Returns:
            List of (timestamp, frame) tuples
        """
        with self._lock:
            frames = list(self._frame_buffer)

        # Return up to n most recent frames
        return frames[-n:] if len(frames) > n else frames

    def is_running(self) -> bool:
        """Check if capture thread is active."""
        return self._running

    def clear_buffer(self) -> None:
        """Clear all frames from buffer."""
        with self._lock:
            self._frame_buffer.clear()

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error

    @property
    def using_fallback(self) -> bool:
        """Check if using fallback video instead of camera."""
        return self._using_fallback

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
