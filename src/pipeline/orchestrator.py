"""
Main orchestrator for the RL meme pipeline.

Connects all components: RL agent, meme player, webcam capture,
emotion analysis, and reward calculation into a unified training loop.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from ..data.template_manager import TemplateManager
from ..data.sound_manager import SoundManager
from ..meme.player import MemePlayer, PlayerConfig
from ..vision.webcam_capture import WebcamCapture, CaptureConfig
from ..vision.emotion_analyzer import EmotionAnalyzer, AnalyzerConfig, EmotionResult
from ..vision.reward_calculator import RewardCalculator, RewardConfig, RewardResult
from ..rl.environment import MemeEnv, EnvConfig
from ..rl.agent import MemeAgent, AgentConfig, create_agent


@dataclass
class OrchestratorConfig:
    """Configuration for the meme orchestrator."""
    
    # Paths
    template_dir: str = "data/templates"
    sound_dir: str = "data/sounds"
    model_path: Optional[str] = None  # Path to load/save model
    
    # Timing
    baseline_duration: float = 3.0  # Seconds to capture baseline
    meme_duration: float = 5.0  # Seconds to show each meme
    reaction_capture_delay: float = 0.5  # Delay before capturing reaction
    inter_meme_pause: float = 1.0  # Pause between memes
    
    # Session
    memes_per_episode: int = 10  # Memes to show per training episode
    save_frequency: int = 5  # Save model every N episodes
    
    # Display
    fullscreen: bool = False
    window_width: int = 800
    window_height: int = 600
    
    # Webcam overlay (PiP)
    show_webcam_overlay: bool = True  # Show webcam PiP during meme display
    pip_width: int = 200  # PiP width in pixels
    pip_height: int = 150  # PiP height in pixels
    pip_position: str = "bottom-right"  # Corner: bottom-right, bottom-left, top-right, top-left
    analyze_every_n_frames: int = 3  # Analyze emotion every Nth frame (performance)
    
    # Webcam
    camera_device: int = 0
    fallback_video: Optional[str] = None
    
    # Training
    learning_rate: float = 3e-4
    exploration_bonus: float = 0.1


@dataclass
class SessionStats:
    """Statistics for a training session."""
    
    total_memes_shown: int = 0
    total_episodes: int = 0
    rewards: List[float] = field(default_factory=list)
    template_usage: Dict[int, int] = field(default_factory=dict)
    sound_usage: Dict[int, int] = field(default_factory=dict)
    best_combinations: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    @property
    def avg_reward(self) -> float:
        """Average reward across all memes."""
        return float(np.mean(self.rewards)) if self.rewards else 0.0
    
    @property
    def recent_avg_reward(self, n: int = 20) -> float:
        """Average of last n rewards."""
        if not self.rewards:
            return 0.0
        recent = self.rewards[-n:]
        return float(np.mean(recent))
    
    @property
    def session_duration(self) -> float:
        """Duration of session in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_memes_shown": self.total_memes_shown,
            "total_episodes": self.total_episodes,
            "avg_reward": self.avg_reward,
            "recent_avg_reward": self.recent_avg_reward,
            "session_duration": self.session_duration,
            "reward_history": self.rewards[-50:],  # Last 50 rewards
            "best_combinations": self.best_combinations[:10],
        }


class MemeOrchestrator:
    """
    Main orchestrator for the RL meme generation pipeline.
    
    Coordinates the full loop:
    1. Agent selects template + sound combination
    2. Display meme with audio
    3. Capture facial reaction during display
    4. Compute reward from reaction
    5. Feed reward back to agent for learning
    
    Features:
    - Baseline capture before each episode
    - Real-time emotion monitoring
    - Model saving/loading
    - Keyboard controls (ESC skip, Q quit)
    - Statistics tracking
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        self.stats = SessionStats()
        
        # Initialize components (lazy)
        self._template_manager: Optional[TemplateManager] = None
        self._sound_manager: Optional[SoundManager] = None
        self._player: Optional[MemePlayer] = None
        self._webcam: Optional[WebcamCapture] = None
        self._analyzer: Optional[EmotionAnalyzer] = None
        self._reward_calc: Optional[RewardCalculator] = None
        self._env: Optional[MemeEnv] = None
        self._agent: Optional[MemeAgent] = None
        
        self._initialized = False
        self._should_quit = False
        
        # Reaction capture during meme display
        self._reaction_frames: List[EmotionResult] = []
        
        # Frame counter for Nth-frame analysis
        self._frame_count: int = 0
        self._last_emotion_result: Optional[EmotionResult] = None
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        logger.info("Initializing MemeOrchestrator...")
        
        try:
            # Template manager
            logger.info("Loading templates...")
            self._template_manager = TemplateManager(self.config.template_dir)
            templates = self._template_manager.templates
            if not templates:
                logger.error(f"No templates found in {self.config.template_dir}")
                return False
            logger.info(f"Loaded {len(templates)} templates")
            
            # Sound manager
            logger.info("Loading sounds...")
            self._sound_manager = SoundManager(self.config.sound_dir)
            sounds = self._sound_manager.available_sounds
            if not sounds:
                logger.error(f"No sounds found in {self.config.sound_dir}")
                return False
            logger.info(f"Loaded {len(sounds)} sounds")
            
            # Meme player
            player_config = PlayerConfig(
                window_width=self.config.window_width,
                window_height=self.config.window_height,
                display_duration=self.config.meme_duration,
                fullscreen=self.config.fullscreen,
            )
            self._player = MemePlayer(player_config)
            
            # Webcam capture
            webcam_config = CaptureConfig(
                device_id=self.config.camera_device,
                fallback_video=self.config.fallback_video,
            )
            self._webcam = WebcamCapture(webcam_config)
            if not self._webcam.start():
                logger.error("Failed to start webcam")
                return False
            logger.info("Webcam started")
            
            # Emotion analyzer
            self._analyzer = EmotionAnalyzer()
            
            # Reward calculator
            self._reward_calc = RewardCalculator()
            
            # RL environment
            env_config = EnvConfig(
                num_templates=len(templates),
                num_sounds=len(sounds),
                max_episode_steps=self.config.memes_per_episode,
                exploration_bonus=self.config.exploration_bonus,
            )
            self._env = MemeEnv(env_config)
            
            # RL agent
            agent_config = AgentConfig(
                learning_rate=self.config.learning_rate,
            )
            self._agent = MemeAgent(env=self._env, config=agent_config)
            
            # Load existing model if specified
            if self.config.model_path and Path(self.config.model_path).exists():
                logger.info(f"Loading model from {self.config.model_path}")
                self._agent.load(self.config.model_path)
            
            self._initialized = True
            logger.info("MemeOrchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self.cleanup()
            return False
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up orchestrator...")
        
        if self._webcam:
            self._webcam.stop()
        
        if self._player:
            self._player.cleanup()
        
        self._initialized = False
    
    def _capture_baseline(self) -> bool:
        """
        Capture baseline emotional state before showing memes.
        
        Returns:
            True if baseline captured successfully
        """
        logger.info(f"Capturing baseline for {self.config.baseline_duration}s...")
        logger.info("Please look at the camera with a neutral expression")
        
        baseline = self._analyzer.capture_baseline(
            self._webcam,
            duration=self.config.baseline_duration,
        )
        
        if not baseline.valid:
            logger.warning("Baseline capture failed - using defaults")
            return False
        
        self._reward_calc.set_baseline(baseline)
        logger.info(f"Baseline captured: avg_amusement={baseline.avg_amusement:.3f}")
        return True
    
    def _create_webcam_pip(
        self,
        webcam_frame: np.ndarray,
        result: Optional[EmotionResult]
    ) -> np.ndarray:
        """
        Create a compact webcam PiP with emotion overlay.
        
        Args:
            webcam_frame: Current webcam frame
            result: Emotion analysis result (may be None)
        
        Returns:
            Resized frame with emotion overlay
        """
        pip_w, pip_h = self.config.pip_width, self.config.pip_height
        pip_frame = cv2.resize(webcam_frame, (pip_w, pip_h))
        
        if result and result.face_detected:
            # Draw face bounding box (scaled to PiP size)
            if result.face_region:
                orig_h, orig_w = webcam_frame.shape[:2]
                scale_x = pip_w / orig_w
                scale_y = pip_h / orig_h
                x = int(result.face_region.get('x', 0) * scale_x)
                y = int(result.face_region.get('y', 0) * scale_y)
                fw = int(result.face_region.get('w', 0) * scale_x)
                fh = int(result.face_region.get('h', 0) * scale_y)
                cv2.rectangle(pip_frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            
            # Dominant emotion label (top-left)
            emotion = result.dominant_emotion or "unknown"
            cv2.putText(
                pip_frame, emotion.upper(),
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
            
            # Amusement bar (bottom)
            amusement = result.amusement_score()
            bar_y = pip_h - 15
            bar_max_width = pip_w - 10
            
            # Background bar
            cv2.rectangle(pip_frame, (5, bar_y), (5 + bar_max_width, bar_y + 10), (50, 50, 50), -1)
            
            # Amusement bar (green if positive, orange if negative)
            bar_color = (0, 255, 0) if amusement >= 0 else (0, 165, 255)
            # Map amusement [-1, 1] to bar width [0, bar_max_width]
            bar_width = int(max(0, min(1, (amusement + 1) / 2)) * bar_max_width)
            cv2.rectangle(pip_frame, (5, bar_y), (5 + bar_width, bar_y + 10), bar_color, -1)
            
            # Center marker (neutral point)
            center_x = 5 + bar_max_width // 2
            cv2.line(pip_frame, (center_x, bar_y - 2), (center_x, bar_y + 12), (255, 255, 255), 1)
        else:
            # No face detected indicator
            cv2.putText(
                pip_frame, "NO FACE",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
            )
        
        # White border around PiP
        cv2.rectangle(pip_frame, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)
        
        return pip_frame
    
    def _add_pip_overlay(
        self,
        main_frame: np.ndarray,
        pip_frame: np.ndarray
    ) -> np.ndarray:
        """
        Composite PiP onto the main frame.
        
        Args:
            main_frame: Main display frame
            pip_frame: PiP webcam frame
        
        Returns:
            Composited frame
        """
        h, w = main_frame.shape[:2]
        pip_h, pip_w = pip_frame.shape[:2]
        margin = 10
        
        # Calculate position based on config
        if self.config.pip_position == "bottom-right":
            x_offset = w - pip_w - margin
            y_offset = h - pip_h - margin
        elif self.config.pip_position == "bottom-left":
            x_offset = margin
            y_offset = h - pip_h - margin
        elif self.config.pip_position == "top-right":
            x_offset = w - pip_w - margin
            y_offset = margin
        elif self.config.pip_position == "top-left":
            x_offset = margin
            y_offset = margin
        else:
            x_offset = w - pip_w - margin
            y_offset = h - pip_h - margin
        
        result = main_frame.copy()
        result[y_offset:y_offset + pip_h, x_offset:x_offset + pip_w] = pip_frame
        return result
    
    def _on_meme_frame(self, frame: np.ndarray, elapsed: float) -> Optional[np.ndarray]:
        """
        Callback for each frame during meme display.
        Captures facial reactions and optionally adds webcam overlay.
        
        Args:
            frame: Current display frame
            elapsed: Seconds since meme started
        
        Returns:
            Modified frame with overlay, or None if no changes
        """
        self._frame_count += 1
        
        if elapsed < self.config.reaction_capture_delay:
            return None  # Skip initial frames, no overlay yet
        
        # Get current webcam frame
        webcam_result = self._webcam.get_frame()
        if webcam_result is None:
            return None
        
        _, webcam_frame = webcam_result
        
        # Analyze emotion every Nth frame for performance
        if self._frame_count % self.config.analyze_every_n_frames == 0:
            result = self._analyzer.analyze_frame(webcam_frame)
            self._last_emotion_result = result
            if result.face_detected:
                self._reaction_frames.append(result)
        
        # Create and add webcam overlay if enabled
        if self.config.show_webcam_overlay:
            pip_frame = self._create_webcam_pip(webcam_frame, self._last_emotion_result)
            return self._add_pip_overlay(frame, pip_frame)
        
        return None
    
    def _show_meme_and_get_reward(
        self,
        template_idx: int,
        sound_idx: int
    ) -> Optional[RewardResult]:
        """
        Show a meme and capture facial reaction for reward.
        
        Args:
            template_idx: Index of template to use
            sound_idx: Index of sound to use
        
        Returns:
            RewardResult if successful, None if skipped/quit
        """
        # Get template and sound
        templates = self._template_manager.templates
        sounds = self._sound_manager.available_sounds
        
        if template_idx >= len(templates) or sound_idx >= len(sounds):
            logger.error(f"Invalid indices: template={template_idx}, sound={sound_idx}")
            return None
        
        template = templates[template_idx]
        sound = sounds[sound_idx]
        
        logger.debug(f"Showing meme: template={template.name}, sound={sound.name}")
        
        # Clear reaction frames and reset frame counter
        self._reaction_frames = []
        self._frame_count = 0
        self._last_emotion_result = None
        
        # Display meme with audio and capture reactions
        # Note: We pass the template image directly since we're not using captions
        completed = self._player.display_image(
            image_path=template.path,
            sound_path=sound.path,
            duration=self.config.meme_duration,
            on_frame=self._on_meme_frame,
        )
        
        if not completed:
            if self._player._should_quit:
                self._should_quit = True
            return None
        
        # Compute reward from reactions
        if not self._reaction_frames:
            logger.warning("No reactions captured - using neutral reward")
            return RewardResult(
                reward=0.0,
                raw_delta=0.0,
                reaction_score=0.0,
                baseline_score=self._reward_calc._baseline.avg_amusement if self._reward_calc._baseline else 0.0,
                timestamp=time.time(),
                face_ratio=0.0,
                meme_id=f"t{template_idx}_s{sound_idx}",
            )
        
        reward_result = self._reward_calc.compute_reward(
            self._reaction_frames,
            meme_id=f"t{template_idx}_s{sound_idx}",
        )
        
        return reward_result
    
    def run_episode(self) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Returns:
            Episode statistics
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "Failed to initialize"}
        
        episode_rewards = []
        episode_start = time.time()
        
        # Capture baseline at start of episode
        self._capture_baseline()
        
        # Reset environment
        obs, info = self._env.reset()
        
        for step in range(self.config.memes_per_episode):
            if self._should_quit:
                break
            
            # Agent selects action
            action = self._agent.select_action(obs)
            template_idx, sound_idx = action
            
            # Show meme and get reward
            reward_result = self._show_meme_and_get_reward(template_idx, sound_idx)
            
            if reward_result is None:
                if self._should_quit:
                    break
                # Skipped - use neutral reward
                reward = 0.0
            else:
                reward = reward_result.reward
                logger.info(f"Step {step+1}: reward={reward:.3f}")
            
            # Set external reward and step environment
            self._env.set_external_reward(reward)
            obs, _, terminated, truncated, info = self._env.step(action)
            
            # Track stats
            episode_rewards.append(reward)
            self.stats.rewards.append(reward)
            self.stats.total_memes_shown += 1
            self.stats.template_usage[template_idx] = self.stats.template_usage.get(template_idx, 0) + 1
            self.stats.sound_usage[sound_idx] = self.stats.sound_usage.get(sound_idx, 0) + 1
            
            # Track best combinations
            if reward > 0.3:  # Good reaction threshold
                self.stats.best_combinations.append({
                    "template_idx": template_idx,
                    "sound_idx": sound_idx,
                    "reward": reward,
                    "timestamp": time.time(),
                })
            
            # Pause between memes
            if step < self.config.memes_per_episode - 1:
                time.sleep(self.config.inter_meme_pause)
            
            if terminated or truncated:
                break
        
        self.stats.total_episodes += 1
        
        return {
            "episode": self.stats.total_episodes,
            "steps": len(episode_rewards),
            "total_reward": sum(episode_rewards),
            "avg_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "duration": time.time() - episode_start,
        }
    
    def train(
        self,
        num_episodes: int = 10,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> SessionStats:
        """
        Run a training session.
        
        Args:
            num_episodes: Number of episodes to run
            callback: Called after each episode with stats
        
        Returns:
            Session statistics
        """
        logger.info(f"Starting training session: {num_episodes} episodes")
        
        if not self.initialize():
            logger.error("Failed to initialize - aborting training")
            return self.stats
        
        try:
            for ep in range(num_episodes):
                if self._should_quit:
                    logger.info("Training stopped by user")
                    break
                
                logger.info(f"\n{'='*50}")
                logger.info(f"Episode {ep + 1}/{num_episodes}")
                logger.info(f"{'='*50}")
                
                episode_stats = self.run_episode()
                
                if callback:
                    callback(episode_stats)
                
                logger.info(
                    f"Episode {ep + 1} complete: "
                    f"reward={episode_stats.get('avg_reward', 0):.3f}, "
                    f"duration={episode_stats.get('duration', 0):.1f}s"
                )
                
                # Save model periodically
                if self.config.model_path and (ep + 1) % self.config.save_frequency == 0:
                    self._agent.save(self.config.model_path)
                    logger.info(f"Model saved to {self.config.model_path}")
            
            # Final save
            if self.config.model_path:
                self._agent.save(self.config.model_path)
                logger.info(f"Final model saved to {self.config.model_path}")
            
        finally:
            self.cleanup()
        
        return self.stats
    
    def demo(self, num_memes: int = 5) -> None:
        """
        Demo mode: Show random memes without training.
        
        Args:
            num_memes: Number of memes to show
        """
        logger.info(f"Demo mode: showing {num_memes} memes")
        
        if not self.initialize():
            logger.error("Failed to initialize")
            return
        
        try:
            self._capture_baseline()
            
            for i in range(num_memes):
                if self._should_quit:
                    break
                
                # Random selection
                templates = self._template_manager.templates
                sounds = self._sound_manager.available_sounds
                template_idx = np.random.randint(0, len(templates))
                sound_idx = np.random.randint(0, len(sounds))
                
                logger.info(f"\nMeme {i + 1}/{num_memes}")
                
                reward_result = self._show_meme_and_get_reward(template_idx, sound_idx)
                
                if reward_result:
                    logger.info(f"Reward: {reward_result.reward:.3f}")
                
                time.sleep(self.config.inter_meme_pause)
            
        finally:
            self.cleanup()
    
    def evaluate(self, num_memes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained agent without updating weights.
        
        Args:
            num_memes: Number of memes to evaluate
        
        Returns:
            Evaluation statistics
        """
        logger.info(f"Evaluation mode: {num_memes} memes")
        
        if not self.initialize():
            return {"error": "Failed to initialize"}
        
        rewards = []
        
        try:
            self._capture_baseline()
            
            obs, _ = self._env.reset()
            
            for i in range(num_memes):
                if self._should_quit:
                    break
                
                # Use agent to select (greedy/deterministic)
                action = self._agent.select_action(obs, deterministic=True)
                template_idx, sound_idx = action
                
                logger.info(f"\nMeme {i + 1}/{num_memes}: template={template_idx}, sound={sound_idx}")
                
                reward_result = self._show_meme_and_get_reward(template_idx, sound_idx)
                
                if reward_result:
                    rewards.append(reward_result.reward)
                    logger.info(f"Reward: {reward_result.reward:.3f}")
                
                # Step environment (for observation update)
                self._env.set_external_reward(reward_result.reward if reward_result else 0.0)
                obs, _, _, _, _ = self._env.step(action)
                
                time.sleep(self.config.inter_meme_pause)
            
        finally:
            self.cleanup()
        
        return {
            "num_memes": len(rewards),
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "min_reward": float(min(rewards)) if rewards else 0.0,
        }
    
    def __enter__(self) -> "MemeOrchestrator":
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
