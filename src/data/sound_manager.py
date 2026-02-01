"""
Sound manager for RL meme generation pipeline.

Provides interface for accessing and selecting sounds.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger


@dataclass
class Sound:
    """A sound effect with metadata."""
    
    name: str
    path: Optional[Path]
    mp3_url: str
    detail_url: str
    downloaded: bool
    
    @property
    def is_available(self) -> bool:
        """Check if the sound file exists locally."""
        return self.path is not None and self.path.exists()
    
    def __repr__(self) -> str:
        status = "✓" if self.is_available else "✗"
        return f"Sound({self.name}, {status})"


class SoundManager:
    """
    Manages sound effects for the RL pipeline.
    
    Provides:
    - Sound discovery from sounds.json metadata
    - Random selection for training
    - Filtering by availability
    - Download status tracking
    """
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.ogg'}
    
    def __init__(self, sounds_dir: str = "data/sounds"):
        """
        Initialize the sound manager.
        
        Args:
            sounds_dir: Directory containing sounds and sounds.json
        """
        self.sounds_dir = Path(sounds_dir)
        self._sounds: List[Sound] = []
        self._metadata_file = self.sounds_dir / "sounds.json"
        self._discovered = False
    
    def _discover_sounds(self) -> None:
        """Load sounds from metadata file and check local files."""
        if self._discovered:
            return
        
        self._sounds = []
        
        # Load metadata if exists
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                for entry in metadata:
                    name = entry.get('name', 'unknown')
                    mp3_url = entry.get('mp3_url', '')
                    detail_url = entry.get('detail_url', '')
                    
                    # Try to find local file
                    local_path = self._find_local_file(name, mp3_url)
                    downloaded = local_path is not None and local_path.exists()
                    
                    sound = Sound(
                        name=name,
                        path=local_path,
                        mp3_url=mp3_url,
                        detail_url=detail_url,
                        downloaded=downloaded
                    )
                    self._sounds.append(sound)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse sounds.json: {e}")
        
        # Also scan for any MP3 files not in metadata
        self._scan_local_files()
        
        self._discovered = True
        available = len([s for s in self._sounds if s.is_available])
        logger.info(f"Discovered {len(self._sounds)} sounds ({available} available locally)")
    
    def _find_local_file(self, name: str, mp3_url: str) -> Optional[Path]:
        """Find the local file for a sound."""
        # Try to extract filename from URL
        if mp3_url:
            url_filename = mp3_url.split('/')[-1]
            local_path = self.sounds_dir / url_filename
            if local_path.exists():
                return local_path
        
        # Try sanitized name
        safe_name = self._sanitize_filename(name)
        for ext in self.SUPPORTED_FORMATS:
            local_path = self.sounds_dir / f"{safe_name}{ext}"
            if local_path.exists():
                return local_path
        
        return None
    
    def _scan_local_files(self) -> None:
        """Scan for local files not in metadata."""
        known_paths = {s.path for s in self._sounds if s.path}
        
        if not self.sounds_dir.exists():
            return
        
        for file_path in self.sounds_dir.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                if file_path not in known_paths:
                    sound = Sound(
                        name=file_path.stem,
                        path=file_path,
                        mp3_url='',
                        detail_url='',
                        downloaded=True
                    )
                    self._sounds.append(sound)
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a name for use as filename."""
        # Replace problematic characters
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            name = name.replace(char, '_')
        return name.strip()
    
    @property
    def sounds(self) -> List[Sound]:
        """Get all sounds (lazy loading)."""
        self._discover_sounds()
        return self._sounds
    
    @property
    def available_sounds(self) -> List[Sound]:
        """Get only sounds that are available locally."""
        return [s for s in self.sounds if s.is_available]
    
    def __len__(self) -> int:
        """Return total number of sounds."""
        return len(self.sounds)
    
    def get_random(self, n: int = 1, available_only: bool = True) -> List[Sound]:
        """
        Get random sounds for RL training.
        
        Args:
            n: Number of sounds to select
            available_only: Only select from locally available sounds
        
        Returns:
            List of randomly selected sounds
        """
        source = self.available_sounds if available_only else self.sounds
        if not source:
            logger.warning("No sounds available for selection")
            return []
        
        n = min(n, len(source))
        return random.sample(source, n)
    
    def get_by_name(self, name: str) -> Optional[Sound]:
        """Get a sound by name (case-insensitive)."""
        name_lower = name.lower()
        for sound in self.sounds:
            if sound.name.lower() == name_lower:
                return sound
        return None
    
    def get_by_index(self, idx: int) -> Optional[Sound]:
        """Get a sound by index."""
        if 0 <= idx < len(self.sounds):
            return self.sounds[idx]
        return None
    
    def refresh(self) -> None:
        """Force re-discovery of sounds."""
        self._discovered = False
        self._discover_sounds()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the sound collection."""
        self._discover_sounds()
        available = [s for s in self._sounds if s.is_available]
        
        return {
            'total': len(self._sounds),
            'available': len(available),
            'missing': len(self._sounds) - len(available),
            'sounds_dir': str(self.sounds_dir),
            'metadata_file': str(self._metadata_file),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"SoundManager({stats['available']}/{stats['total']} sounds available)"
