#!/usr/bin/env python3
"""
CLI script for testing webcam capture and emotion detection.

Usage:
    python scripts/test_webcam.py --mode preview      # Show webcam with emotion overlay
    python scripts/test_webcam.py --mode baseline     # Capture baseline emotions
    python scripts/test_webcam.py --mode full-test    # Complete pipeline test
    python scripts/test_webcam.py --video-file x.mp4  # Use video instead of camera
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from loguru import logger

from src.vision import (
    WebcamCapture,
    CaptureConfig,
    EmotionAnalyzer,
    AnalyzerConfig,
    RewardCalculator,
    RewardConfig,
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level
    )


def draw_emotion_overlay(
    frame: np.ndarray,
    emotions: dict,
    dominant: str | None,
    amusement: float,
    face_region: dict | None = None
) -> np.ndarray:
    """Draw emotion information overlay on frame."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    
    # Draw face bounding box if detected
    if face_region:
        x, y = face_region.get('x', 0), face_region.get('y', 0)
        fw, fh = face_region.get('w', 0), face_region.get('h', 0)
        cv2.rectangle(overlay, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
    
    # Draw semi-transparent background for text
    cv2.rectangle(overlay, (10, 10), (280, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
    
    # Draw title
    cv2.putText(overlay, "Emotion Analysis", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw dominant emotion
    if dominant:
        color = (0, 255, 0) if dominant in ['happy', 'surprise'] else (255, 255, 255)
        cv2.putText(overlay, f"Dominant: {dominant}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        cv2.putText(overlay, "No face detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw amusement score bar
    cv2.putText(overlay, f"Amusement: {amusement:.3f}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw amusement bar
    bar_width = int(max(0, min(1, amusement + 0.5)) * 200)  # Normalize -0.5 to 0.5 -> 0 to 200
    bar_color = (0, 255, 0) if amusement > 0 else (0, 165, 255)
    cv2.rectangle(overlay, (20, 95), (20 + bar_width, 110), bar_color, -1)
    cv2.rectangle(overlay, (20, 95), (220, 110), (255, 255, 255), 1)
    
    # Draw emotion breakdown
    y_offset = 130
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:4]
    for emotion, score in top_emotions:
        bar_w = int(score * 1.5)  # Scale: 100% -> 150px
        cv2.putText(overlay, f"{emotion[:7]:7s}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(overlay, (90, y_offset - 10), (90 + bar_w, y_offset), (100, 200, 100), -1)
        cv2.putText(overlay, f"{score:.0f}%", (95 + bar_w, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y_offset += 18
    
    return overlay


def run_preview_mode(webcam: WebcamCapture, analyzer: EmotionAnalyzer):
    """Run preview mode with live emotion detection overlay."""
    logger.info("Starting preview mode. Press 'q' to quit.")
    
    frame_count = 0
    last_result = None
    analysis_interval = 5  # Analyze every N frames (DeepFace is slow)
    
    while True:
        frame_data = webcam.get_frame(timeout=1.0)
        
        if frame_data is None:
            logger.warning("No frame available")
            time.sleep(0.1)
            continue
        
        timestamp, frame = frame_data
        frame_count += 1
        
        # Analyze periodically (not every frame - too slow)
        if frame_count % analysis_interval == 0 or last_result is None:
            last_result = analyzer.analyze_frame(frame, timestamp)
        
        # Draw overlay
        if last_result and last_result.face_detected:
            overlay = draw_emotion_overlay(
                frame,
                last_result.emotions,
                last_result.dominant_emotion,
                last_result.amusement_score(),
                last_result.face_region
            )
        else:
            overlay = draw_emotion_overlay(frame, {}, None, 0.0)
        
        # Show frame
        cv2.imshow("Webcam Preview - Press 'q' to quit", overlay)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    logger.info("Preview mode ended")


def run_baseline_mode(webcam: WebcamCapture, analyzer: EmotionAnalyzer, duration: float = 5.0):
    """Capture and display baseline emotional state."""
    logger.info(f"Capturing baseline for {duration} seconds...")
    logger.info("Please maintain a neutral expression.")
    
    # Show countdown
    for i in range(3, 0, -1):
        frame_data = webcam.get_frame()
        if frame_data:
            _, frame = frame_data
            cv2.putText(frame, f"Starting in {i}...", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow("Baseline Capture", frame)
            cv2.waitKey(1000)
    
    # Capture baseline
    baseline = analyzer.capture_baseline(webcam, duration=duration)
    
    cv2.destroyAllWindows()
    
    # Display results
    if baseline.valid:
        logger.info("=" * 50)
        logger.info("BASELINE RESULTS")
        logger.info("=" * 50)
        logger.info(f"Frames captured: {baseline.frame_count}")
        logger.info(f"Average amusement: {baseline.avg_amusement:.4f}")
        logger.info("Average emotions:")
        for emotion, score in sorted(baseline.avg_emotions.items(), key=lambda x: -x[1]):
            logger.info(f"  {emotion}: {score:.2f}%")
    else:
        logger.error(f"Baseline capture failed: {baseline.error}")
    
    return baseline


def run_full_test(webcam: WebcamCapture, analyzer: EmotionAnalyzer, calculator: RewardCalculator):
    """Run full pipeline test: baseline -> stimulus -> measure reaction -> compute reward."""
    logger.info("=" * 50)
    logger.info("FULL PIPELINE TEST")
    logger.info("=" * 50)
    
    # Step 1: Capture baseline
    logger.info("\n[Step 1] Capturing baseline (neutral expression)...")
    baseline = run_baseline_mode(webcam, analyzer, duration=3.0)
    
    if not baseline.valid:
        logger.error("Cannot proceed without valid baseline")
        return
    
    calculator.set_baseline(baseline)
    
    # Step 2: Show "stimulus" (in real use, this would be a meme)
    logger.info("\n[Step 2] Showing stimulus...")
    logger.info("Try to SMILE or LAUGH for the next 5 seconds!")
    
    # Countdown
    for i in range(3, 0, -1):
        frame_data = webcam.get_frame()
        if frame_data:
            _, frame = frame_data
            cv2.putText(frame, f"Smile in {i}...", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            cv2.imshow("Reaction Test", frame)
            cv2.waitKey(1000)
    
    # Step 3: Capture reaction
    logger.info("\n[Step 3] Capturing reaction...")
    reaction_results = []
    start_time = time.time()
    reaction_duration = 5.0
    
    while time.time() - start_time < reaction_duration:
        frame_data = webcam.get_frame()
        if frame_data:
            timestamp, frame = frame_data
            result = analyzer.analyze_frame(frame, timestamp)
            reaction_results.append(result)
            
            # Show live feedback
            if result.face_detected:
                overlay = draw_emotion_overlay(
                    frame, result.emotions, result.dominant_emotion,
                    result.amusement_score(), result.face_region
                )
            else:
                overlay = frame.copy()
            
            elapsed = time.time() - start_time
            cv2.putText(overlay, f"Recording: {elapsed:.1f}s / {reaction_duration}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Reaction Test", overlay)
            cv2.waitKey(1)
        
        time.sleep(0.3)  # Sample interval
    
    cv2.destroyAllWindows()
    
    # Step 4: Compute reward
    logger.info("\n[Step 4] Computing reward...")
    reward_result = calculator.compute_reward(reaction_results, meme_id="test_meme")
    
    # Display results
    logger.info("=" * 50)
    logger.info("REWARD RESULTS")
    logger.info("=" * 50)
    logger.info(f"Baseline amusement:  {reward_result.baseline_score:.4f}")
    logger.info(f"Reaction amusement:  {reward_result.reaction_score:.4f}")
    logger.info(f"Raw delta:           {reward_result.raw_delta:.4f}")
    logger.info(f"Face detection rate: {reward_result.face_ratio:.1%}")
    logger.info("-" * 50)
    logger.info(f"FINAL REWARD:        {reward_result.reward:.4f}")
    logger.info("=" * 50)
    
    if reward_result.reward > 0.3:
        logger.info("🎉 Great reaction! The meme made you happy!")
    elif reward_result.reward > 0:
        logger.info("🙂 Slight positive reaction detected.")
    elif reward_result.reward < -0.3:
        logger.info("😐 Negative reaction - meme wasn't funny.")
    else:
        logger.info("😶 Neutral reaction.")


def main():
    parser = argparse.ArgumentParser(
        description="Test webcam capture and emotion detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  preview    - Live webcam feed with emotion detection overlay
  baseline   - Capture baseline emotional state (neutral expression)
  full-test  - Complete pipeline: baseline -> stimulus -> reaction -> reward

Examples:
  # Live preview with emotion overlay
  python scripts/test_webcam.py --mode preview

  # Capture baseline emotions
  python scripts/test_webcam.py --mode baseline

  # Full pipeline test
  python scripts/test_webcam.py --mode full-test

  # Use video file instead of camera (WSL2 fallback)
  python scripts/test_webcam.py --mode preview --video-file data/test_video.mp4
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preview", "baseline", "full-test"],
        default="preview",
        help="Test mode to run (default: preview)"
    )
    
    parser.add_argument(
        "--video-file",
        type=str,
        default=None,
        help="Path to video file to use instead of camera (for WSL2 testing)"
    )
    
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    
    parser.add_argument(
        "--detector",
        type=str,
        choices=["opencv", "ssd", "mtcnn", "retinaface"],
        default="opencv",
        help="Face detector backend (default: opencv)"
    )
    
    parser.add_argument(
        "--baseline-duration",
        type=float,
        default=5.0,
        help="Duration in seconds for baseline capture (default: 5.0)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Configure capture
    capture_config = CaptureConfig(
        device_id=args.camera_id,
        fallback_video=args.video_file
    )
    
    # Configure analyzer
    analyzer_config = AnalyzerConfig(
        detector_backend=args.detector
    )
    
    # Configure reward calculator
    reward_config = RewardConfig()
    
    logger.info("Initializing webcam capture...")
    
    with WebcamCapture(capture_config) as webcam:
        if not webcam.is_running():
            logger.error(f"Failed to start webcam: {webcam.last_error}")
            if not args.video_file:
                logger.info("Tip: Use --video-file to test with a video file instead")
            return 1
        
        if webcam.using_fallback:
            logger.info("Using fallback video file")
        
        analyzer = EmotionAnalyzer(analyzer_config)
        calculator = RewardCalculator(reward_config)
        
        logger.info("Loading DeepFace model (first time may take a while)...")
        
        if args.mode == "preview":
            run_preview_mode(webcam, analyzer)
        elif args.mode == "baseline":
            run_baseline_mode(webcam, analyzer, duration=args.baseline_duration)
        elif args.mode == "full-test":
            run_full_test(webcam, analyzer, calculator)
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
