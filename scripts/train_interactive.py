#!/usr/bin/env python3
"""
Interactive training script for the RL meme pipeline.

This script runs the full training loop:
1. Captures your baseline facial expression
2. Shows memes with audio
3. Monitors your facial reactions
4. Uses reactions as RL rewards
5. Learns your humor preferences over time

Usage:
    # Demo mode (no training, random memes)
    python scripts/train_interactive.py --mode demo --num-memes 5

    # Training mode
    python scripts/train_interactive.py --mode train --episodes 10

    # Evaluate trained model
    python scripts/train_interactive.py --mode eval --model data/models/meme_agent.zip

    # With custom settings
    python scripts/train_interactive.py --mode train --fullscreen --meme-duration 6
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.pipeline import MemeOrchestrator, OrchestratorConfig


def setup_logger(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive RL meme training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Demo (random memes):  python scripts/train_interactive.py --mode demo
  Train 10 episodes:    python scripts/train_interactive.py --mode train -e 10
  Evaluate model:       python scripts/train_interactive.py --mode eval --model data/models/agent.zip
  Fullscreen training:  python scripts/train_interactive.py --mode train --fullscreen
        """
    )
    
    # Mode
    parser.add_argument(
        "--mode", "-m",
        choices=["demo", "train", "eval"],
        default="demo",
        help="Operation mode (default: demo)"
    )
    
    # Training settings
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=5,
        help="Number of training episodes (default: 5)"
    )
    parser.add_argument(
        "--memes-per-episode",
        type=int,
        default=10,
        help="Memes shown per episode (default: 10)"
    )
    parser.add_argument(
        "--num-memes", "-n",
        type=int,
        default=5,
        help="Number of memes for demo/eval mode (default: 5)"
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to load/save model"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5,
        help="Save model every N episodes (default: 5)"
    )
    
    # Display settings
    parser.add_argument(
        "--fullscreen", "-f",
        action="store_true",
        help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=800,
        help="Window width (default: 800)"
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=600,
        help="Window height (default: 600)"
    )
    
    # Timing
    parser.add_argument(
        "--meme-duration",
        type=float,
        default=5.0,
        help="Seconds to show each meme (default: 5.0)"
    )
    parser.add_argument(
        "--baseline-duration",
        type=float,
        default=3.0,
        help="Seconds to capture baseline (default: 3.0)"
    )
    
    # Webcam
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--fallback-video",
        type=str,
        default=None,
        help="Video file for testing without camera"
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable webcam PiP overlay"
    )
    parser.add_argument(
        "--pip-size",
        type=str,
        default="200x150",
        help="Webcam overlay size WxH (default: 200x150)"
    )
    parser.add_argument(
        "--pip-position",
        choices=["bottom-right", "bottom-left", "top-right", "top-left"],
        default="bottom-right",
        help="Webcam overlay position (default: bottom-right)"
    )
    
    # Data paths
    parser.add_argument(
        "--templates",
        type=str,
        default="data/templates",
        help="Template directory (default: data/templates)"
    )
    parser.add_argument(
        "--sounds",
        type=str,
        default="data/sounds",
        help="Sound directory (default: data/sounds)"
    )
    
    # Other
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    setup_logger(args.verbose)
    
    # Parse PiP size
    try:
        pip_w, pip_h = map(int, args.pip_size.split('x'))
    except ValueError:
        logger.error(f"Invalid --pip-size format: {args.pip_size}. Use WxH (e.g., 200x150)")
        return 1
    
    # Build config
    config = OrchestratorConfig(
        template_dir=args.templates,
        sound_dir=args.sounds,
        model_path=args.model,
        baseline_duration=args.baseline_duration,
        meme_duration=args.meme_duration,
        memes_per_episode=args.memes_per_episode,
        save_frequency=args.save_freq,
        fullscreen=args.fullscreen,
        window_width=args.window_width,
        window_height=args.window_height,
        camera_device=args.camera,
        fallback_video=args.fallback_video,
        show_webcam_overlay=not args.no_overlay,
        pip_width=pip_w,
        pip_height=pip_h,
        pip_position=args.pip_position,
    )
    
    # Create orchestrator
    orchestrator = MemeOrchestrator(config)
    
    logger.info("=" * 60)
    logger.info("RL Meme Generator - Interactive Training")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Templates: {args.templates}")
    logger.info(f"Sounds: {args.sounds}")
    if args.model:
        logger.info(f"Model: {args.model}")
    logger.info("")
    logger.info("Controls:")
    logger.info("  ESC - Skip current meme")
    logger.info("  Q   - Quit training")
    logger.info("=" * 60)
    
    try:
        if args.mode == "demo":
            logger.info("\nStarting demo mode...")
            logger.info("Watch the memes and react naturally!")
            orchestrator.demo(num_memes=args.num_memes)
            
        elif args.mode == "train":
            logger.info(f"\nStarting training: {args.episodes} episodes")
            logger.info("Your reactions will be used to train the model!")
            
            def on_episode(stats):
                logger.info(
                    f"Episode {stats['episode']}: "
                    f"avg_reward={stats['avg_reward']:.3f}, "
                    f"duration={stats['duration']:.1f}s"
                )
            
            final_stats = orchestrator.train(
                num_episodes=args.episodes,
                callback=on_episode,
            )
            
            logger.info("\n" + "=" * 60)
            logger.info("Training Complete!")
            logger.info("=" * 60)
            logger.info(f"Total memes shown: {final_stats.total_memes_shown}")
            logger.info(f"Total episodes: {final_stats.total_episodes}")
            logger.info(f"Average reward: {final_stats.avg_reward:.3f}")
            logger.info(f"Session duration: {final_stats.session_duration:.1f}s")
            
            if final_stats.best_combinations:
                logger.info("\nTop combinations (highest rewards):")
                for i, combo in enumerate(final_stats.best_combinations[:5]):
                    logger.info(
                        f"  {i+1}. Template {combo['template_idx']}, "
                        f"Sound {combo['sound_idx']}: reward={combo['reward']:.3f}"
                    )
            
        elif args.mode == "eval":
            if not args.model:
                logger.error("--model required for eval mode")
                return 1
            
            logger.info(f"\nEvaluating model: {args.model}")
            eval_stats = orchestrator.evaluate(num_memes=args.num_memes)
            
            logger.info("\n" + "=" * 60)
            logger.info("Evaluation Complete!")
            logger.info("=" * 60)
            logger.info(f"Memes evaluated: {eval_stats['num_memes']}")
            logger.info(f"Average reward: {eval_stats['avg_reward']:.3f}")
            logger.info(f"Std deviation: {eval_stats['std_reward']:.3f}")
            logger.info(f"Best reward: {eval_stats['max_reward']:.3f}")
            logger.info(f"Worst reward: {eval_stats['min_reward']:.3f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
