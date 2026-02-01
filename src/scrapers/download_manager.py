"""
Download manager for MP3 sound files.

Handles concurrent downloads with retry logic and progress tracking.
"""

import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from loguru import logger

from .sound_scraper import SoundInfo


@dataclass
class DownloadResult:
    """Result of a download attempt."""
    sound: SoundInfo
    success: bool
    filepath: Optional[str] = None
    error: Optional[str] = None


class SoundDownloadManager:
    """
    Manages downloading of sound files.

    Features:
    - Concurrent downloads with configurable workers
    - Retry logic with exponential backoff
    - Progress tracking
    - Skip existing files
    """

    def __init__(
        self,
        output_dir: str = "data/sounds",
        max_workers: int = 5,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the download manager.

        Args:
            output_dir: Directory to save downloaded files
            max_workers: Maximum concurrent downloads
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts per file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "audio/mpeg, audio/*, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })

        return session

    @staticmethod
    def sanitize_filename(name: str, max_length: int = 100) -> str:
        """
        Sanitize a string to be safe for use as a filename.

        Args:
            name: Original name
            max_length: Maximum filename length

        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")

        # Replace multiple spaces/underscores with single underscore
        name = re.sub(r"[\s_]+", "_", name)

        # Remove leading/trailing underscores and dots
        name = name.strip("_.")

        # Truncate to max length
        if len(name) > max_length:
            name = name[:max_length].rstrip("_")

        return name or "unnamed"

    def get_filepath(self, sound: SoundInfo) -> Path:
        """Get the local filepath for a sound."""
        safe_name = self.sanitize_filename(sound.name)
        return self.output_dir / f"{safe_name}.mp3"

    def download_sound(
        self,
        sound: SoundInfo,
        skip_existing: bool = True
    ) -> DownloadResult:
        """
        Download a single sound file.

        Args:
            sound: SoundInfo with mp3_url set
            skip_existing: Skip if file already exists

        Returns:
            DownloadResult with success status
        """
        if not sound.mp3_url:
            return DownloadResult(
                sound=sound,
                success=False,
                error="No MP3 URL available"
            )

        filepath = self.get_filepath(sound)

        # Skip if exists
        if skip_existing and filepath.exists():
            logger.debug(f"Skipping existing: {filepath.name}")
            sound.downloaded = True
            return DownloadResult(
                sound=sound,
                success=True,
                filepath=str(filepath)
            )

        try:
            response = self.session.get(
                sound.mp3_url,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Verify it's actually audio content
            content_type = response.headers.get("Content-Type", "")
            if "audio" not in content_type and "octet-stream" not in content_type:
                logger.warning(f"Unexpected content type for {sound.name}: {content_type}")

            # Write to file
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify file was written
            if filepath.exists() and filepath.stat().st_size > 0:
                sound.downloaded = True
                logger.debug(f"Downloaded: {filepath.name}")
                return DownloadResult(
                    sound=sound,
                    success=True,
                    filepath=str(filepath)
                )
            else:
                return DownloadResult(
                    sound=sound,
                    success=False,
                    error="File is empty or not created"
                )

        except requests.exceptions.Timeout:
            return DownloadResult(
                sound=sound,
                success=False,
                error="Request timeout"
            )
        except requests.exceptions.RequestException as e:
            return DownloadResult(
                sound=sound,
                success=False,
                error=str(e)
            )
        except IOError as e:
            return DownloadResult(
                sound=sound,
                success=False,
                error=f"IO Error: {e}"
            )

    def download_all(
        self,
        sounds: List[SoundInfo],
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, int]:
        """
        Download all sounds concurrently.

        Args:
            sounds: List of SoundInfo objects to download
            skip_existing: Skip files that already exist
            progress_callback: Optional callback(completed, total)

        Returns:
            Dictionary with download statistics
        """
        stats = {
            "total": len(sounds),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "no_url": 0
        }

        # Filter sounds with URLs
        to_download = [s for s in sounds if s.mp3_url]
        stats["no_url"] = len(sounds) - len(to_download)

        if not to_download:
            logger.warning("No sounds with MP3 URLs to download")
            return stats

        logger.info(f"Downloading {len(to_download)} sounds with {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_sound, sound, skip_existing): sound
                for sound in to_download
            }

            with tqdm(total=len(to_download), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result.success:
                        # Check if it was actually downloaded or skipped
                        if result.filepath and Path(result.filepath).exists():
                            file_age = time.time() - Path(result.filepath).stat().st_mtime
                            if file_age < 5:  # Downloaded in last 5 seconds
                                stats["success"] += 1
                            else:
                                stats["skipped"] += 1
                        else:
                            stats["success"] += 1
                    else:
                        stats["failed"] += 1
                        logger.warning(f"Failed: {result.sound.name} - {result.error}")

                    pbar.update(1)

                    if progress_callback:
                        completed = stats["success"] + stats["failed"] + stats["skipped"]
                        progress_callback(completed, len(to_download))

        logger.info(
            f"Download complete: {stats['success']} succeeded, "
            f"{stats['skipped']} skipped, {stats['failed']} failed, "
            f"{stats['no_url']} had no URL"
        )

        return stats

    def get_downloaded_sounds(self) -> List[Path]:
        """Get list of all downloaded sound files."""
        return list(self.output_dir.glob("*.mp3"))

    def get_download_count(self) -> int:
        """Get count of downloaded files."""
        return len(self.get_downloaded_sounds())

    def clear_downloads(self):
        """Remove all downloaded files."""
        for f in self.get_downloaded_sounds():
            f.unlink()
        logger.info(f"Cleared all downloads from {self.output_dir}")
