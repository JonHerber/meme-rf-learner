"""
Google Drive folder downloader for meme templates.

Uses gdown for simple public folder downloads without OAuth complexity.
"""

import re
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict
import json

import gdown
from loguru import logger


@dataclass
class TemplateInfo:
    """Information about a downloaded template."""
    filename: str
    filepath: str
    file_id: Optional[str] = None
    size_bytes: int = 0
    downloaded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class DriveLoader:
    """
    Downloads meme templates from a public Google Drive folder.

    Features:
    - Extract folder ID from various URL formats
    - Download all images from folder
    - Skip existing files for resume capability
    - Track download metadata
    """

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

    def __init__(
        self,
        output_dir: str = "data/templates",
        quiet: bool = False
    ):
        """
        Initialize the drive loader.

        Args:
            output_dir: Directory to save downloaded templates
            quiet: Suppress gdown progress output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        self.templates: List[TemplateInfo] = []

    @staticmethod
    def extract_folder_id(url_or_id: str) -> str:
        """
        Extract folder ID from Google Drive URL or return ID if already extracted.

        Handles formats:
        - https://drive.google.com/drive/folders/FOLDER_ID
        - https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing
        - https://drive.google.com/drive/u/0/folders/FOLDER_ID
        - FOLDER_ID (raw ID)

        Args:
            url_or_id: Google Drive folder URL or folder ID

        Returns:
            Extracted folder ID

        Raises:
            ValueError: If folder ID cannot be extracted
        """
        # Check if it's already a raw ID (no slashes, no dots)
        if re.match(r'^[\w-]{20,}$', url_or_id):
            return url_or_id

        # Try to extract from URL
        patterns = [
            r'/folders/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)

        raise ValueError(f"Could not extract folder ID from: {url_or_id}")

    def _get_existing_files(self) -> set:
        """Get set of existing filenames in output directory."""
        existing = set()
        for ext in self.SUPPORTED_FORMATS:
            for f in self.output_dir.glob(f"*{ext}"):
                existing.add(f.name)
            for f in self.output_dir.glob(f"*{ext.upper()}"):
                existing.add(f.name)
        return existing

    def download_folder(
        self,
        folder_url_or_id: str,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Download all images from a Google Drive folder.

        Args:
            folder_url_or_id: Google Drive folder URL or ID
            skip_existing: Skip files that already exist locally

        Returns:
            Dictionary with download statistics
        """
        stats = {
            "total": 0,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "non_image": 0
        }

        folder_id = self.extract_folder_id(folder_url_or_id)
        logger.info(f"Downloading from folder ID: {folder_id}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")

        # Get existing files for skip logic
        existing_files = set()
        if skip_existing:
            existing_files = self._get_existing_files()
            if existing_files:
                logger.info(f"Found {len(existing_files)} existing files")

        try:
            # gdown.download_folder returns list of downloaded file paths
            downloaded_files = gdown.download_folder(
                id=folder_id,
                output=str(self.output_dir),
                quiet=self.quiet,
                remaining_ok=True  # Continue even if some files fail
            )

            if downloaded_files is None:
                logger.error("Failed to download folder - check if folder is public")
                stats["failed"] = 1
                return stats

            # Process results
            for filepath_str in downloaded_files:
                filepath = Path(filepath_str)
                stats["total"] += 1

                # Check if it's an image
                if filepath.suffix.lower() not in self.SUPPORTED_FORMATS:
                    stats["non_image"] += 1
                    logger.debug(f"Non-image file: {filepath.name}")
                    continue

                # Track template info
                try:
                    size = filepath.stat().st_size if filepath.exists() else 0
                except OSError:
                    size = 0

                template = TemplateInfo(
                    filename=filepath.name,
                    filepath=str(filepath.absolute()),
                    size_bytes=size,
                    downloaded=True
                )
                self.templates.append(template)

                if filepath.name in existing_files:
                    stats["skipped"] += 1
                else:
                    stats["downloaded"] += 1

        except Exception as e:
            logger.error(f"Download failed: {e}")
            stats["failed"] += 1

        logger.info(
            f"Download complete: {stats['downloaded']} new, "
            f"{stats['skipped']} skipped, {stats['non_image']} non-image, "
            f"{stats['failed']} failed"
        )

        return stats

    def get_downloaded_templates(self) -> List[Path]:
        """Get list of all template files in output directory."""
        templates = []
        for ext in self.SUPPORTED_FORMATS:
            templates.extend(self.output_dir.glob(f"*{ext}"))
            templates.extend(self.output_dir.glob(f"*{ext.upper()}"))
        return sorted(templates)

    def get_template_count(self) -> int:
        """Get count of downloaded template files."""
        return len(self.get_downloaded_templates())

    def save_metadata(self, filepath: Optional[str] = None):
        """Save template metadata to JSON file."""
        if filepath is None:
            filepath = str(self.output_dir / "templates.json")

        data = [t.to_dict() for t in self.templates]
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.templates)} template records to {filepath}")

    @classmethod
    def load_metadata(cls, filepath: str) -> List[TemplateInfo]:
        """Load template metadata from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [TemplateInfo(**item) for item in data]
