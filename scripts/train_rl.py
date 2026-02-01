#!/usr/bin/env python3
"""
Training script for the meme RL agent.

Modes:
- simulate: Train with simulated random rewards (for testing)
- interactive: Train with real facial reaction rewards
- eval: Evaluate a trained model
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from loguru import logger

from src.rl import MemeEnv, EnvConfig, MemeAgent, AgentConfig, create_agent


def setup_logging(verbose: bool) -> None:
    """Configure logging level."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<level>{message}</level>")


def train_simulated(
    num_templates: int,
    num_sounds: int,
    total_timesteps: int,
    save_path: str,
    load_path: str = None
) -> None:
    """
    Train with simulated rewards (for testing the RL pipeline).
    
    Uses random rewards to verify the training loop works.
    """
    logger.info("Starting simulated training")
    logger.info(f"Templates: {num_templates}, Sounds: {num_sounds}")
    logger.info(f"Total timesteps: {total_timesteps}")
    
    # Create agent
    agent = create_agent(
        num_templates=num_templates,
        num_sounds=num_sounds,
        model_path=load_path,
        learning_rate=3e-4,
        ent_coef=0.05,  # Encourage exploration
    )
    
    # Train
    agent.learn(total_timesteps=total_timesteps)
    
    # Save model
    if save_path:
        agent.save(save_path)
    
    # Print stats
    stats = agent.get_stats()
    logger.info("Training complete!")
    logger.info(f"Environment stats: {stats['env']}")
    logger.info(f"Training stats: {stats['training']}")


def evaluate_model(
    model_path: str,
    num_templates: int,
    num_sounds: int,
    num_episodes: int = 10
) -> None:
    """Evaluate a trained model."""
    logger.info(f"Evaluating model: {model_path}")
    
    # Load agent
    agent = create_agent(
        num_templates=num_templates,
        num_sounds=num_sounds,
        model_path=model_path
    )
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs = agent.reset_env()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Get action (deterministic for evaluation)
            template_idx, sound_idx = agent.get_action(deterministic=True)
            
            # Simulate step (in real use, this would be from facial reaction)
            agent.env.set_reward(np.random.uniform(-0.5, 0.5))
            obs, reward, terminated, truncated, info = agent.env.step(
                np.array([template_idx, sound_idx])
            )
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        logger.info(f"Episode {ep + 1}: Reward = {episode_reward:.3f}, Steps = {steps}")
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  Mean Reward: {np.mean(episode_rewards):.3f}")
    logger.info(f"  Std Reward: {np.std(episode_rewards):.3f}")
    logger.info(f"  Min Reward: {np.min(episode_rewards):.3f}")
    logger.info(f"  Max Reward: {np.max(episode_rewards):.3f}")


def show_environment_info(num_templates: int, num_sounds: int) -> None:
    """Show information about the environment."""
    env_config = EnvConfig(num_templates=num_templates, num_sounds=num_sounds)
    env = MemeEnv(config=env_config)
    
    print("\n=== Meme RL Environment ===\n")
    print(f"Templates: {num_templates}")
    print(f"Sounds: {num_sounds}")
    print(f"Possible combinations: {num_templates * num_sounds:,}")
    print(f"\nObservation space: {env.observation_space}")
    print(f"  - One-hot template: {num_templates} dims")
    print(f"  - One-hot sound: {num_sounds} dims")
    print(f"  - Reward history: {env_config.reward_history_size} dims")
    print(f"  - Total: {env.observation_space.shape[0]} dims")
    print(f"\nAction space: {env.action_space}")
    print(f"  - Action[0]: template index (0-{num_templates-1})")
    print(f"  - Action[1]: sound index (0-{num_sounds-1})")
    print(f"\nMax episode steps: {env_config.max_episode_steps}")
    print(f"Exploration bonus: {env_config.exploration_bonus}")
    print(f"Repeat penalty: {env_config.repeat_penalty}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train the meme RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show environment info
  python scripts/train_rl.py --mode info
  
  # Train with simulated rewards
  python scripts/train_rl.py --mode simulate --timesteps 10000 --save model.zip
  
  # Evaluate a trained model
  python scripts/train_rl.py --mode eval --load model.zip
  
  # Continue training from checkpoint
  python scripts/train_rl.py --mode simulate --load model.zip --save model_v2.zip
"""
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["simulate", "interactive", "eval", "info"],
        default="info",
        help="Operation mode (default: info)"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=10000,
        help="Total training timesteps (default: 10000)"
    )
    parser.add_argument(
        "--num-templates",
        type=int,
        default=75,
        help="Number of templates (default: 75)"
    )
    parser.add_argument(
        "--num-sounds",
        type=int,
        default=194,
        help="Number of sounds (default: 194)"
    )
    parser.add_argument(
        "--save", "-s",
        default="data/models/meme_agent.zip",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--load", "-l",
        default=None,
        help="Path to load existing model"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=10,
        help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        if args.mode == "info":
            show_environment_info(args.num_templates, args.num_sounds)
        
        elif args.mode == "simulate":
            train_simulated(
                num_templates=args.num_templates,
                num_sounds=args.num_sounds,
                total_timesteps=args.timesteps,
                save_path=args.save,
                load_path=args.load
            )
        
        elif args.mode == "eval":
            if not args.load:
                logger.error("--load required for eval mode")
                sys.exit(1)
            evaluate_model(
                model_path=args.load,
                num_templates=args.num_templates,
                num_sounds=args.num_sounds,
                num_episodes=args.episodes
            )
        
        elif args.mode == "interactive":
            logger.error("Interactive mode not yet implemented - use Phase 6 orchestrator")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
