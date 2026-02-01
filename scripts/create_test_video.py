#!/usr/bin/env python3
"""
Create a synthetic test video for webcam testing in WSL2.

This generates a simple video with a face-like pattern that can be used
to test the emotion detection pipeline when no camera is available.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def create_synthetic_face_frame(
    width: int = 640,
    height: int = 480,
    frame_num: int = 0,
    expression: str = "neutral"
) -> np.ndarray:
    """
    Create a frame with a simple synthetic face pattern.
    
    Args:
        width: Frame width
        height: Frame height
        frame_num: Current frame number (for animation)
        expression: One of 'neutral', 'happy', 'surprised'
    
    Returns:
        BGR image as numpy array
    """
    # Create blank frame with skin-tone-ish background
    frame = np.full((height, width, 3), (200, 180, 160), dtype=np.uint8)
    
    # Face center
    cx, cy = width // 2, height // 2
    
    # Draw face oval
    face_color = (180, 160, 140)
    cv2.ellipse(frame, (cx, cy), (120, 150), 0, 0, 360, face_color, -1)
    
    # Eyes
    eye_y = cy - 30
    left_eye_x = cx - 50
    right_eye_x = cx + 50
    
    # Eye whites
    cv2.ellipse(frame, (left_eye_x, eye_y), (25, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, (right_eye_x, eye_y), (25, 15), 0, 0, 360, (255, 255, 255), -1)
    
    # Pupils (animate slightly)
    pupil_offset_x = int(5 * np.sin(frame_num * 0.1))
    cv2.circle(frame, (left_eye_x + pupil_offset_x, eye_y), 8, (50, 50, 50), -1)
    cv2.circle(frame, (right_eye_x + pupil_offset_x, eye_y), 8, (50, 50, 50), -1)
    
    # Eyebrows
    brow_y = eye_y - 25
    if expression == "surprised":
        brow_y -= 10  # Raised eyebrows
    cv2.line(frame, (left_eye_x - 20, brow_y), (left_eye_x + 20, brow_y - 5), (80, 60, 40), 3)
    cv2.line(frame, (right_eye_x - 20, brow_y - 5), (right_eye_x + 20, brow_y), (80, 60, 40), 3)
    
    # Nose
    nose_y = cy + 20
    cv2.line(frame, (cx, cy - 10), (cx - 5, nose_y), (150, 130, 110), 2)
    cv2.line(frame, (cx - 5, nose_y), (cx + 5, nose_y), (150, 130, 110), 2)
    
    # Mouth - varies by expression
    mouth_y = cy + 60
    if expression == "happy":
        # Smiling mouth
        cv2.ellipse(frame, (cx, mouth_y - 10), (40, 25), 0, 20, 160, (100, 80, 80), 3)
    elif expression == "surprised":
        # Open mouth (O shape)
        cv2.ellipse(frame, (cx, mouth_y), (20, 30), 0, 0, 360, (100, 80, 80), 3)
    else:
        # Neutral mouth
        cv2.line(frame, (cx - 30, mouth_y), (cx + 30, mouth_y), (100, 80, 80), 3)
    
    return frame


def create_test_video(
    output_path: str,
    duration: float = 10.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480
) -> None:
    """
    Create a test video with varying expressions.
    
    Args:
        output_path: Path to save the video
        duration: Video duration in seconds
        fps: Frames per second
        width: Frame width
        height: Frame height
    """
    total_frames = int(duration * fps)
    
    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        sys.exit(1)
    
    print(f"Creating test video: {output_path}")
    print(f"  Duration: {duration}s, FPS: {fps}, Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    
    # Expression schedule: neutral -> happy -> neutral -> surprised -> neutral
    expressions = [
        (0.0, 0.3, "neutral"),
        (0.3, 0.5, "happy"),
        (0.5, 0.7, "neutral"),
        (0.7, 0.85, "surprised"),
        (0.85, 1.0, "neutral"),
    ]
    
    for i in range(total_frames):
        progress = i / total_frames
        
        # Find current expression
        expression = "neutral"
        for start, end, expr in expressions:
            if start <= progress < end:
                expression = expr
                break
        
        frame = create_synthetic_face_frame(width, height, i, expression)
        
        # Add text overlay showing expression
        cv2.putText(
            frame, f"Expression: {expression}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        cv2.putText(
            frame, f"Frame: {i}/{total_frames}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
        
        out.write(frame)
        
        if (i + 1) % (fps * 2) == 0:
            print(f"  Progress: {i + 1}/{total_frames} frames ({100 * (i + 1) / total_frames:.0f}%)")
    
    out.release()
    print(f"✓ Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a synthetic test video for webcam testing"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/test_video.mp4",
        help="Output video path (default: data/test_video.mp4)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Video duration in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_test_video(
        str(output_path),
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()
