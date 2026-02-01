"""
PPO agent wrapper for meme generation RL.

Uses stable-baselines3 for the PPO implementation with
custom configuration for handling noisy human rewards.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
from loguru import logger

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not installed - agent training disabled")

from .environment import MemeEnv, EnvConfig


@dataclass
class AgentConfig:
    """Configuration for the PPO agent."""
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 64  # Steps per update (smaller for interactive training)
    batch_size: int = 32
    n_epochs: int = 5
    gamma: float = 0.95  # Discount factor (lower for immediate rewards)
    gae_lambda: float = 0.9
    clip_range: float = 0.2
    ent_coef: float = 0.05  # Higher entropy for exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network architecture
    policy_type: str = "MlpPolicy"
    net_arch: tuple = (64, 64)  # Hidden layer sizes
    
    # Training settings
    verbose: int = 1
    device: str = "auto"


class TrainingCallback(BaseCallback):
    """Callback for logging training progress."""
    
    def __init__(self, log_interval: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self._last_buffer_len = 0
    
    def _on_step(self) -> bool:
        # Check for new episode completions
        try:
            buffer = self.model.ep_info_buffer
            if buffer is not None and hasattr(buffer, '__len__'):
                current_len = len(buffer)
                if current_len > self._last_buffer_len:
                    # New episodes completed
                    for i in range(self._last_buffer_len, current_len):
                        ep_info = buffer[i]
                        if isinstance(ep_info, dict):
                            self.episode_rewards.append(ep_info.get('r', 0))
                            self.episode_lengths.append(ep_info.get('l', 0))
                    self._last_buffer_len = current_len
        except Exception:
            pass  # Silently ignore buffer access errors
        
        if self.n_calls % self.log_interval == 0 and self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])
            logger.info(
                f"Step {self.n_calls}: "
                f"Episodes = {len(self.episode_rewards)}, "
                f"Avg Reward (last 10) = {avg_reward:.3f}"
            )
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "recent_avg_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
            "avg_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
        }


class MemeAgent:
    """
    PPO-based agent for learning meme preferences.
    
    Features:
    - Wrapped stable-baselines3 PPO
    - Save/load model checkpoints
    - Configurable hyperparameters for noisy rewards
    - Action prediction with exploration
    """
    
    def __init__(
        self,
        env: Optional[MemeEnv] = None,
        config: Optional[AgentConfig] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the agent.
        
        Args:
            env: MemeEnv instance (created if None)
            config: Agent configuration
            model_path: Path to load existing model
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. Install with: "
                "pip install stable-baselines3"
            )
        
        self.config = config or AgentConfig()
        self.env = env or MemeEnv()
        self.model: Optional[PPO] = None
        self.callback = TrainingCallback()
        
        if model_path:
            self.load(model_path)
        else:
            self._create_model()
    
    def _create_model(self) -> None:
        """Create a new PPO model."""
        policy_kwargs = {
            "net_arch": list(self.config.net_arch)
        }
        
        self.model = PPO(
            policy=self.config.policy_type,
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=self.config.verbose,
            device=self.config.device,
        )
        
        logger.info("Created new PPO model")
    
    def predict(
        self,
        observation: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> tuple:
        """
        Predict the next action.
        
        Args:
            observation: Current observation (uses env's if None)
            deterministic: If True, use greedy action selection
        
        Returns:
            Tuple of (action, model_state)
        """
        if observation is None:
            observation = self.env._get_obs()
        
        action, state = self.model.predict(
            observation,
            deterministic=deterministic
        )
        
        return action, state
    
    def get_action(self, deterministic: bool = False) -> tuple:
        """
        Get action indices for template and sound.
        
        Args:
            deterministic: If True, use greedy selection
        
        Returns:
            Tuple of (template_idx, sound_idx)
        """
        action, _ = self.predict(deterministic=deterministic)
        return int(action[0]), int(action[1])
    
    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> tuple:
        """
        Select action given an observation.
        
        Args:
            observation: Current state observation
            deterministic: If True, use greedy action selection
        
        Returns:
            Tuple of (template_idx, sound_idx)
        """
        action, _ = self.predict(observation, deterministic=deterministic)
        return int(action[0]), int(action[1])
    
    def learn(
        self,
        total_timesteps: int = 1000,
        callback: Optional[BaseCallback] = None,
        progress_bar: bool = True
    ) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training steps
            callback: Optional custom callback
            progress_bar: Show progress bar
        """
        cb = callback or self.callback
        
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=cb,
            progress_bar=progress_bar
        )
        
        logger.info("Training complete")
    
    def train_step(self) -> Dict[str, Any]:
        """
        Perform a single training step.
        
        Used for interactive training where rewards come from
        real-time facial reactions.
        
        Returns:
            Info dict from the environment step
        """
        # Get current observation
        obs = self.env._get_obs()
        
        # Predict action
        action, _ = self.model.predict(obs, deterministic=False)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        return info
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load a model from disk."""
        path = Path(path)
        if not path.exists():
            # Try with .zip extension
            if not path.with_suffix('.zip').exists():
                raise FileNotFoundError(f"Model not found: {path}")
            path = path.with_suffix('.zip')
        
        self.model = PPO.load(str(path), env=self.env)
        logger.info(f"Model loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        try:
            env_stats = self.env.get_stats() if hasattr(self.env, 'get_stats') else {}
        except Exception:
            env_stats = {}
        
        try:
            training_stats = self.callback.get_stats() if self.callback else {}
        except Exception:
            training_stats = {}
        
        return {
            "env": env_stats,
            "training": training_stats,
        }
    
    def reset_env(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        obs, _ = self.env.reset()
        return obs


def create_agent(
    num_templates: int = 75,
    num_sounds: int = 194,
    model_path: Optional[str] = None,
    **kwargs
) -> MemeAgent:
    """
    Factory function to create a configured agent.
    
    Args:
        num_templates: Number of available templates
        num_sounds: Number of available sounds
        model_path: Path to load existing model
        **kwargs: Additional AgentConfig parameters
    
    Returns:
        Configured MemeAgent instance
    """
    env_config = EnvConfig(
        num_templates=num_templates,
        num_sounds=num_sounds
    )
    env = MemeEnv(config=env_config)
    
    agent_config = AgentConfig(**kwargs)
    
    return MemeAgent(
        env=env,
        config=agent_config,
        model_path=model_path
    )
