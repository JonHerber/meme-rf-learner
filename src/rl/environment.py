"""
Custom Gymnasium environment for meme generation RL.

The agent learns to select template + sound combinations
that maximize user amusement (facial reaction rewards).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger


@dataclass
class EnvConfig:
    """Configuration for the meme RL environment."""
    
    num_templates: int = 75  # Number of available templates
    num_sounds: int = 194  # Number of available sounds
    reward_history_size: int = 10  # Recent rewards to include in state
    max_episode_steps: int = 50  # Max memes per episode
    
    # Reward shaping
    exploration_bonus: float = 0.1  # Bonus for trying new combinations
    repeat_penalty: float = -0.05  # Penalty for repeating recent combos


class MemeEnv(gym.Env):
    """
    Gymnasium environment for meme selection.
    
    Observation Space:
        - One-hot encoded last template selection
        - One-hot encoded last sound selection  
        - Recent reward history (normalized)
    
    Action Space:
        MultiDiscrete([num_templates, num_sounds])
        - Action[0]: template index
        - Action[1]: sound index
    
    Reward:
        Provided externally from facial reaction analysis.
        Range: [-1, 1] (clipped amusement delta)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the meme environment.
        
        Args:
            config: Environment configuration
            render_mode: Rendering mode (unused for now)
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # Action space: [template_idx, sound_idx]
        self.action_space = spaces.MultiDiscrete([
            self.config.num_templates,
            self.config.num_sounds
        ])
        
        # Observation space components
        obs_dim = (
            self.config.num_templates +  # One-hot template
            self.config.num_sounds +  # One-hot sound
            self.config.reward_history_size  # Recent rewards
        )
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Internal state
        self._last_template_idx: int = 0
        self._last_sound_idx: int = 0
        self._reward_history: List[float] = []
        self._step_count: int = 0
        self._episode_count: int = 0
        self._combination_counts: Dict[Tuple[int, int], int] = {}
        self._recent_combinations: List[Tuple[int, int]] = []
        
        # External reward (set by orchestrator after showing meme)
        self._pending_reward: Optional[float] = None
        
        logger.debug(
            f"MemeEnv initialized: {self.config.num_templates} templates, "
            f"{self.config.num_sounds} sounds, obs_dim={obs_dim}"
        )
    
    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        # One-hot encode last template
        template_one_hot = np.zeros(self.config.num_templates, dtype=np.float32)
        template_one_hot[self._last_template_idx] = 1.0
        
        # One-hot encode last sound
        sound_one_hot = np.zeros(self.config.num_sounds, dtype=np.float32)
        sound_one_hot[self._last_sound_idx] = 1.0
        
        # Reward history (padded with zeros if needed)
        reward_hist = np.zeros(self.config.reward_history_size, dtype=np.float32)
        for i, r in enumerate(self._reward_history[-self.config.reward_history_size:]):
            reward_hist[i] = r
        
        return np.concatenate([template_one_hot, sound_one_hot, reward_hist])
    
    def _get_info(self) -> Dict[str, Any]:
        """Build the info dictionary."""
        avg_reward = float(np.mean(self._reward_history)) if self._reward_history else 0.0
        return {
            "step": int(self._step_count),
            "episode_num": int(self._episode_count),  # Renamed to avoid conflict with Monitor
            "last_template": int(self._last_template_idx),
            "last_sound": int(self._last_sound_idx),
            "total_combinations_tried": int(len(self._combination_counts)),
            "avg_reward": avg_reward,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self._last_template_idx = self.np_random.integers(0, self.config.num_templates)
        self._last_sound_idx = self.np_random.integers(0, self.config.num_sounds)
        self._reward_history = []
        self._step_count = 0
        self._episode_count += 1
        self._recent_combinations = []
        self._pending_reward = None
        
        # Keep combination_counts across episodes for exploration tracking
        
        logger.debug(f"Episode {self._episode_count} started")
        
        return self._get_obs(), self._get_info()
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Note: The reward should be set via `set_reward()` before calling
        step() for accurate reward assignment.
        
        Args:
            action: [template_idx, sound_idx]
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        template_idx = int(action[0])
        sound_idx = int(action[1])
        
        # Validate actions
        template_idx = np.clip(template_idx, 0, self.config.num_templates - 1)
        sound_idx = np.clip(sound_idx, 0, self.config.num_sounds - 1)
        
        # Update state
        self._last_template_idx = template_idx
        self._last_sound_idx = sound_idx
        self._step_count += 1
        
        # Track combination
        combo = (template_idx, sound_idx)
        self._combination_counts[combo] = self._combination_counts.get(combo, 0) + 1
        self._recent_combinations.append(combo)
        if len(self._recent_combinations) > 10:
            self._recent_combinations.pop(0)
        
        # Get reward (from external source or pending)
        if self._pending_reward is not None:
            base_reward = self._pending_reward
            self._pending_reward = None
        else:
            # No reward yet - will be 0 (placeholder)
            base_reward = 0.0
        
        # Apply reward shaping
        reward = self._shape_reward(base_reward, combo)
        self._reward_history.append(reward)
        
        # Check termination
        terminated = False
        truncated = self._step_count >= self.config.max_episode_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _shape_reward(self, base_reward: float, combo: Tuple[int, int]) -> float:
        """Apply reward shaping for exploration."""
        reward = base_reward
        
        # Exploration bonus for new combinations
        if self._combination_counts.get(combo, 0) == 1:
            reward += self.config.exploration_bonus
        
        # Penalty for repeating recent combinations
        if combo in self._recent_combinations[:-1]:  # Exclude current
            reward += self.config.repeat_penalty
        
        return float(np.clip(reward, -1.0, 1.0))
    
    def set_reward(self, reward: float) -> None:
        """
        Set the reward for the next step.
        
        Called by the orchestrator after showing the meme and
        capturing the facial reaction.
        
        Args:
            reward: Reward value from facial reaction analysis
        """
        self._pending_reward = float(np.clip(reward, -1.0, 1.0))
    
    def set_external_reward(self, reward: float) -> None:
        """
        Alias for set_reward() for orchestrator compatibility.
        
        Args:
            reward: Reward value from facial reaction analysis
        """
        self.set_reward(reward)
    
    def get_action_indices(self) -> Tuple[int, int]:
        """Get the current action indices for external use."""
        return self._last_template_idx, self._last_sound_idx
    
    def render(self) -> None:
        """Render the environment (logging only)."""
        if self.render_mode == "human":
            logger.info(
                f"Step {self._step_count}: "
                f"Template={self._last_template_idx}, "
                f"Sound={self._last_sound_idx}, "
                f"Avg Reward={np.mean(self._reward_history) if self._reward_history else 0:.3f}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_episodes": self._episode_count,
            "total_steps": self._step_count,
            "unique_combinations": len(self._combination_counts),
            "possible_combinations": self.config.num_templates * self.config.num_sounds,
            "coverage": len(self._combination_counts) / max(1, self.config.num_templates * self.config.num_sounds),
            "avg_reward": np.mean(self._reward_history) if self._reward_history else 0.0,
            "reward_std": np.std(self._reward_history) if len(self._reward_history) > 1 else 0.0,
            "top_combinations": self._get_top_combinations(5),
        }
    
    def _get_top_combinations(self, n: int) -> List[Dict[str, Any]]:
        """Get the most frequently used combinations."""
        sorted_combos = sorted(
            self._combination_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {"template": c[0], "sound": c[1], "count": count}
            for (c, count) in sorted_combos[:n]
        ]
    
    def get_combination_count(self, template_idx: int, sound_idx: int) -> int:
        """Get how many times a specific combination was used."""
        return self._combination_counts.get((template_idx, sound_idx), 0)
