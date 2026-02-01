"""
Meme player module for displaying memes with audio.

Provides combined display with OpenCV window and pygame audio playback.
"""

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Callable

import cv2
import numpy as np
from loguru import logger

# Try to import pygame for audio
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("pygame not installed - audio playback disabled")


@dataclass
class PlayerConfig:
    """Configuration for meme player."""
    
    window_name: str = "Meme Generator"
    window_width: int = 800
    window_height: int = 600
    display_duration: float = 5.0  # Seconds to display meme
    audio_volume: float = 0.7  # 0.0 to 1.0
    fade_duration: float = 0.3  # Fade in/out duration
    fullscreen: bool = False
    audio_enabled: bool = True


class AudioPlayer:
    """
    Pygame-based audio player for sound effects.
    
    Features:
    - Non-blocking audio playback
    - Volume control
    - Support for MP3, WAV, OGG
    """
    
    def __init__(self, volume: float = 0.7):
        """
        Initialize the audio player.
        
        Args:
            volume: Initial volume (0.0 to 1.0)
        """
        self._initialized = False
        self._volume = volume
        self._current_sound: Optional[pygame.mixer.Sound] = None
        
        if PYGAME_AVAILABLE:
            self._init_pygame()
    
    def _init_pygame(self) -> None:
        """Initialize pygame mixer."""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(8)
            self._initialized = True
            logger.debug("Pygame mixer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            self._initialized = False
    
    @property
    def is_available(self) -> bool:
        """Check if audio playback is available."""
        return PYGAME_AVAILABLE and self._initialized
    
    def play(
        self,
        sound_path: Union[str, Path],
        volume: Optional[float] = None,
        blocking: bool = False
    ) -> bool:
        """
        Play a sound file.
        
        Args:
            sound_path: Path to the sound file
            volume: Override volume (0.0 to 1.0)
            blocking: If True, wait for playback to complete
        
        Returns:
            True if playback started successfully
        """
        if not self.is_available:
            logger.warning("Audio playback not available")
            return False
        
        sound_path = Path(sound_path)
        if not sound_path.exists():
            logger.warning(f"Sound file not found: {sound_path}")
            return False
        
        try:
            self._current_sound = pygame.mixer.Sound(str(sound_path))
            vol = volume if volume is not None else self._volume
            self._current_sound.set_volume(vol)
            self._current_sound.play()
            
            if blocking:
                # Wait for playback to complete
                while pygame.mixer.get_busy():
                    time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to play sound {sound_path}: {e}")
            return False
    
    def stop(self) -> None:
        """Stop current playback."""
        if self._initialized:
            pygame.mixer.stop()
    
    def set_volume(self, volume: float) -> None:
        """Set playback volume."""
        self._volume = max(0.0, min(1.0, volume))
        if self._current_sound:
            self._current_sound.set_volume(self._volume)
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if not self._initialized:
            return False
        return pygame.mixer.get_busy()
    
    def get_duration(self, sound_path: Union[str, Path]) -> float:
        """Get duration of a sound file in seconds."""
        if not self.is_available:
            return 0.0
        
        try:
            sound = pygame.mixer.Sound(str(sound_path))
            return sound.get_length()
        except Exception:
            return 0.0
    
    def cleanup(self) -> None:
        """Clean up pygame resources."""
        if self._initialized:
            pygame.mixer.quit()
            self._initialized = False


class MemePlayer:
    """
    Combined meme display with audio playback.
    
    Features:
    - OpenCV window for image display
    - Pygame audio playback
    - Synchronized display and audio
    - Keyboard controls (ESC to skip, Q to quit)
    """
    
    def __init__(self, config: Optional[PlayerConfig] = None):
        """
        Initialize the meme player.
        
        Args:
            config: Player configuration
        """
        self.config = config or PlayerConfig()
        self.audio = AudioPlayer(volume=self.config.audio_volume)
        self._window_created = False
        self._should_quit = False
    
    def _create_window(self) -> None:
        """Create the display window."""
        if self._window_created:
            return
        
        if self.config.fullscreen:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.config.window_name,
                self.config.window_width,
                self.config.window_height
            )
        
        self._window_created = True
    
    def _resize_for_display(self, image: np.ndarray) -> np.ndarray:
        """Resize image to fit display window while maintaining aspect ratio."""
        h, w = image.shape[:2]
        target_w = self.config.window_width
        target_h = self.config.window_height
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _apply_fade(self, image: np.ndarray, alpha: float) -> np.ndarray:
        """Apply fade effect to image."""
        if alpha >= 1.0:
            return image
        return (image * alpha).astype(np.uint8)
    
    def display(
        self,
        meme,  # ComposedMeme
        sound_path: Optional[Union[str, Path]] = None,
        duration: Optional[float] = None,
        on_frame: Optional[Callable[[np.ndarray, float], None]] = None
    ) -> bool:
        """
        Display a meme with optional audio.
        
        Args:
            meme: ComposedMeme object to display
            sound_path: Path to sound file to play
            duration: Override display duration
            on_frame: Callback for each frame (image, elapsed_time)
        
        Returns:
            True if completed normally, False if skipped/quit
        """
        self._create_window()
        
        # Get display duration
        display_time = duration or self.config.display_duration
        if sound_path and self.audio.is_available:
            sound_duration = self.audio.get_duration(sound_path)
            if sound_duration > 0:
                display_time = max(display_time, sound_duration + 0.5)
        
        # Convert meme to OpenCV format
        if hasattr(meme, 'to_numpy'):
            frame = meme.to_numpy()
        else:
            frame = np.array(meme)
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = self._resize_for_display(frame)
        
        # Start audio playback
        if sound_path and self.config.audio_enabled:
            self.audio.play(sound_path, blocking=False)
        
        start_time = time.time()
        fade_in_end = start_time + self.config.fade_duration
        fade_out_start = start_time + display_time - self.config.fade_duration
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed >= display_time:
                break
            
            # Calculate fade alpha
            if time.time() < fade_in_end:
                alpha = (time.time() - start_time) / self.config.fade_duration
            elif time.time() > fade_out_start:
                alpha = (start_time + display_time - time.time()) / self.config.fade_duration
            else:
                alpha = 1.0
            
            # Apply fade and display
            display_frame = self._apply_fade(frame, min(1.0, max(0.0, alpha)))
            
            # Callback for frame processing (e.g., emotion capture, overlay)
            # Callback can return modified frame for display
            if on_frame:
                modified = on_frame(display_frame, elapsed)
                if modified is not None:
                    display_frame = modified
            
            cv2.imshow(self.config.window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(33) & 0xFF  # ~30 FPS
            if key == 27:  # ESC - skip this meme
                self.audio.stop()
                return False
            elif key == ord('q') or key == ord('Q'):  # Q - quit
                self._should_quit = True
                self.audio.stop()
                return False
        
        return True
    
    def display_image(
        self,
        image: Optional[Union[np.ndarray, "Image.Image"]] = None,
        image_path: Optional[Union[str, Path]] = None,
        sound_path: Optional[Union[str, Path]] = None,
        duration: Optional[float] = None,
        on_frame: Optional[Callable[[np.ndarray, float], None]] = None
    ) -> bool:
        """
        Display an image with optional audio.
        
        Args:
            image: Image to display (numpy array or PIL Image)
            image_path: Path to image file (alternative to image)
            sound_path: Path to sound file to play
            duration: Display duration in seconds
            on_frame: Callback for each frame (image, elapsed_time)
        
        Returns:
            True if completed, False if skipped/quit
        """
        self._create_window()
        
        # Load image from path if provided
        if image is None and image_path:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return False
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return False
        elif image is None:
            logger.warning("No image or image_path provided")
            return False
        
        # Convert PIL to numpy if needed
        if hasattr(image, 'convert'):  # PIL Image
            image = np.array(image.convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        image = self._resize_for_display(image)
        
        # Determine display duration
        display_time = duration or self.config.display_duration
        if sound_path and self.audio.is_available:
            sound_duration = self.audio.get_duration(sound_path)
            if sound_duration > 0:
                display_time = max(display_time, sound_duration + 0.5)
        
        # Start audio playback
        if sound_path and self.config.audio_enabled:
            self.audio.play(sound_path, blocking=False)
        
        start_time = time.time()
        fade_in_end = start_time + self.config.fade_duration
        fade_out_start = start_time + display_time - self.config.fade_duration
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed >= display_time:
                break
            
            # Calculate fade alpha
            if time.time() < fade_in_end:
                alpha = (time.time() - start_time) / self.config.fade_duration
            elif time.time() > fade_out_start:
                alpha = (start_time + display_time - time.time()) / self.config.fade_duration
            else:
                alpha = 1.0
            
            # Apply fade and display
            display_frame = self._apply_fade(image, min(1.0, max(0.0, alpha)))
            
            # Callback for frame processing (e.g., emotion capture, overlay)
            # Callback can return modified frame for display
            if on_frame:
                modified = on_frame(display_frame, elapsed)
                if modified is not None:
                    display_frame = modified
            
            cv2.imshow(self.config.window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(33) & 0xFF
            if key == 27:  # ESC - skip
                self.audio.stop()
                return False
            elif key == ord('q') or key == ord('Q'):
                self._should_quit = True
                self.audio.stop()
                return False
        
        return True
    
    def show_countdown(self, seconds: int = 3, message: str = "Get ready!") -> None:
        """
        Show a countdown before meme display.
        
        Args:
            seconds: Countdown duration
            message: Message to display
        """
        self._create_window()
        
        for i in range(seconds, 0, -1):
            # Create countdown frame
            frame = np.zeros(
                (self.config.window_height, self.config.window_width, 3),
                dtype=np.uint8
            )
            
            # Add text
            cv2.putText(
                frame, message,
                (50, self.config.window_height // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2
            )
            cv2.putText(
                frame, str(i),
                (self.config.window_width // 2 - 30, self.config.window_height // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 4
            )
            
            cv2.imshow(self.config.window_name, frame)
            cv2.waitKey(1000)
    
    def show_message(
        self,
        message: str,
        duration: float = 2.0,
        color: tuple = (255, 255, 255)
    ) -> None:
        """Display a simple text message."""
        self._create_window()
        
        frame = np.zeros(
            (self.config.window_height, self.config.window_width, 3),
            dtype=np.uint8
        )
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(message, font, scale, thickness)
        x = (self.config.window_width - text_w) // 2
        y = (self.config.window_height + text_h) // 2
        
        cv2.putText(frame, message, (x, y), font, scale, color, thickness)
        
        cv2.imshow(self.config.window_name, frame)
        cv2.waitKey(int(duration * 1000))
    
    @property
    def should_quit(self) -> bool:
        """Check if user requested quit."""
        return self._should_quit
    
    def reset_quit(self) -> None:
        """Reset the quit flag."""
        self._should_quit = False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.audio.cleanup()
        cv2.destroyAllWindows()
        self._window_created = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()
