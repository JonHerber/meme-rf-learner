"""
Template manager for RL meme generation pipeline.

Provides interface for accessing and selecting meme templates.
"""

import random
from pathlib import Path
from typing import List, Optional, Dict

from PIL import Image
from loguru import logger


class Template:
    """A meme template with metadata."""

    def __init__(self, path: Path, width: int, height: int, img_format: str):
        self.path = path
        self.name = path.stem
        self.width = width
        self.height = height
        self.format = img_format

    @property
    def aspect_ratio(self) -> float:
        """Width to height ratio."""
        return self.width / self.height if self.height > 0 else 1.0

    def load_image(self) -> Image.Image:
        """Load and return the PIL Image."""
        return Image.open(self.path)

    def __repr__(self) -> str:
        return f"Template({self.name}, {self.width}x{self.height})"


class TemplateManager:
    """
    Manages meme templates for the RL pipeline.

    Provides:
    - Template discovery and loading
    - Random selection for training
    - Filtering by attributes (size, format, aspect ratio)
    - Caching for performance
    """

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

    def __init__(
        self,
        template_dir: str = "data/templates",
        lazy_load: bool = True
    ):
        """
        Initialize the template manager.

        Args:
            template_dir: Directory containing template images
            lazy_load: If True, only load metadata on demand
        """
        self.template_dir = Path(template_dir)
        self.lazy_load = lazy_load
        self._templates: Optional[List[Template]] = None
        self._template_cache: Dict[str, Template] = {}

        if not self.template_dir.exists():
            logger.warning(f"Template directory does not exist: {self.template_dir}")

    @property
    def templates(self) -> List[Template]:
        """Get all templates (loads on first access if lazy_load=True)."""
        if self._templates is None:
            self._templates = self._discover_templates()
        return self._templates

    def _discover_templates(self) -> List[Template]:
        """Scan directory and load template metadata."""
        templates = []

        if not self.template_dir.exists():
            logger.warning(f"Template directory not found: {self.template_dir}")
            return templates

        for ext in self.SUPPORTED_FORMATS:
            # Use recursive glob to find templates in subdirectories
            for path in self.template_dir.glob(f"**/*{ext}"):
                try:
                    template = self._load_template_metadata(path)
                    templates.append(template)
                except Exception as e:
                    logger.warning(f"Failed to load template {path.name}: {e}")

            # Also check uppercase extensions
            for path in self.template_dir.glob(f"**/*{ext.upper()}"):
                try:
                    template = self._load_template_metadata(path)
                    templates.append(template)
                except Exception as e:
                    logger.warning(f"Failed to load template {path.name}: {e}")

        logger.info(f"Discovered {len(templates)} templates in {self.template_dir}")
        return templates

    def _load_template_metadata(self, path: Path) -> Template:
        """Load metadata for a single template."""
        # Check cache first
        cache_key = str(path)
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        # Load image to get dimensions
        with Image.open(path) as img:
            width, height = img.size
            img_format = img.format or path.suffix[1:].upper()

        template = Template(
            path=path,
            width=width,
            height=height,
            img_format=img_format
        )

        self._template_cache[cache_key] = template
        return template

    def get_random(self, n: int = 1) -> List[Template]:
        """
        Get n random templates for training.

        Args:
            n: Number of templates to return

        Returns:
            List of random Template objects
        """
        if not self.templates:
            raise ValueError("No templates available")

        n = min(n, len(self.templates))
        return random.sample(self.templates, n)

    def get_by_name(self, name: str) -> Optional[Template]:
        """Get template by name (filename without extension)."""
        for template in self.templates:
            if template.name == name:
                return template
        return None

    def get_by_index(self, index: int) -> Template:
        """Get template by index."""
        return self.templates[index]

    def filter_by_size(
        self,
        min_width: int = 0,
        min_height: int = 0,
        max_width: int = 999999,
        max_height: int = 999999
    ) -> List[Template]:
        """Filter templates by dimensions."""
        return [
            t for t in self.templates
            if min_width <= t.width <= max_width
            and min_height <= t.height <= max_height
        ]

    def filter_by_aspect_ratio(
        self,
        min_ratio: float = 0.0,
        max_ratio: float = 999.0
    ) -> List[Template]:
        """Filter templates by aspect ratio (width/height)."""
        return [
            t for t in self.templates
            if min_ratio <= t.aspect_ratio <= max_ratio
        ]

    def filter_by_format(self, formats: List[str]) -> List[Template]:
        """Filter templates by image format."""
        formats_upper = {f.upper().lstrip('.') for f in formats}
        return [
            t for t in self.templates
            if t.format.upper() in formats_upper
        ]

    def list_names(self) -> List[str]:
        """Get list of all template names."""
        return [t.name for t in self.templates]

    def refresh(self):
        """Re-scan directory for new templates."""
        self._templates = None
        self._template_cache.clear()
        _ = self.templates  # Trigger reload

    def __len__(self) -> int:
        return len(self.templates)

    def __iter__(self):
        return iter(self.templates)

    def __getitem__(self, index: int) -> Template:
        return self.templates[index]
