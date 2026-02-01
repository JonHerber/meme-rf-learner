"""
Meme composer module for combining templates with text.

Provides PIL-based text overlay functionality for meme generation.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


@dataclass
class TextStyle:
    """Configuration for meme text styling."""
    
    font_path: Optional[str] = None  # Path to TTF font, None for default
    font_size: int = 48
    font_color: Tuple[int, int, int] = (255, 255, 255)  # White
    stroke_color: Tuple[int, int, int] = (0, 0, 0)  # Black outline
    stroke_width: int = 3
    uppercase: bool = True
    max_width_ratio: float = 0.9  # Max text width as ratio of image width
    padding: int = 20  # Padding from edges


@dataclass
class TextPosition:
    """Position for text overlay."""
    
    TOP = "top"
    BOTTOM = "bottom"
    CENTER = "center"
    
    position: str = "bottom"  # top, bottom, or center
    y_offset: int = 0  # Additional offset from calculated position


@dataclass
class MemeConfig:
    """Configuration for meme generation."""
    
    top_text_style: TextStyle = field(default_factory=TextStyle)
    bottom_text_style: TextStyle = field(default_factory=TextStyle)
    output_format: str = "RGB"  # RGB or RGBA
    max_output_size: Tuple[int, int] = (1920, 1080)  # Max dimensions
    maintain_aspect_ratio: bool = True


@dataclass
class ComposedMeme:
    """A composed meme ready for display."""
    
    image: Image.Image
    template_name: str
    top_text: Optional[str]
    bottom_text: Optional[str]
    sound_name: Optional[str] = None
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (BGR for OpenCV)."""
        rgb_array = np.array(self.image.convert('RGB'))
        # Convert RGB to BGR for OpenCV
        return rgb_array[:, :, ::-1]
    
    def save(self, path: Union[str, Path], quality: int = 95) -> None:
        """Save the composed meme to a file."""
        self.image.save(str(path), quality=quality)
    
    def __repr__(self) -> str:
        return f"ComposedMeme({self.template_name}, top='{self.top_text}', bottom='{self.bottom_text}')"


class MemeComposer:
    """
    Composes memes by overlaying text on templates.
    
    Features:
    - PIL-based text rendering with outline
    - Automatic text wrapping and sizing
    - Support for top and bottom captions
    - Configurable styling
    """
    
    # Default captions for variety
    DEFAULT_CAPTIONS = [
        "ME WHEN",
        "POV:",
        "NOBODY:",
        "LITERALLY NO ONE:",
        "WHEN YOU",
        "THAT MOMENT WHEN",
        "HOW IT FEELS TO",
        "WHEN THE",
        "BE LIKE",
        "SOCIETY IF",
    ]
    
    def __init__(self, config: Optional[MemeConfig] = None):
        """
        Initialize the meme composer.
        
        Args:
            config: Meme generation configuration
        """
        self.config = config or MemeConfig()
        self._font_cache: dict = {}
    
    def _get_font(self, style: TextStyle, size: Optional[int] = None) -> ImageFont.FreeTypeFont:
        """Get or create a font with caching."""
        font_size = size or style.font_size
        cache_key = (style.font_path, font_size)
        
        if cache_key not in self._font_cache:
            try:
                if style.font_path and Path(style.font_path).exists():
                    font = ImageFont.truetype(style.font_path, font_size)
                else:
                    # Try common system fonts
                    for font_name in ['Impact', 'DejaVuSans-Bold', 'Arial-Bold', 'FreeSansBold']:
                        try:
                            font = ImageFont.truetype(font_name, font_size)
                            break
                        except OSError:
                            continue
                    else:
                        # Fall back to default font
                        font = ImageFont.load_default()
                        logger.warning("Using default font - consider installing Impact font")
            except Exception as e:
                logger.warning(f"Font loading failed: {e}, using default")
                font = ImageFont.load_default()
            
            self._font_cache[cache_key] = font
        
        return self._font_cache[cache_key]
    
    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int
    ) -> List[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def _calculate_font_size(
        self,
        text: str,
        style: TextStyle,
        max_width: int,
        max_height: int
    ) -> int:
        """Calculate optimal font size for text to fit."""
        font_size = style.font_size
        min_size = 12
        
        while font_size > min_size:
            font = self._get_font(style, font_size)
            lines = self._wrap_text(text, font, max_width)
            
            # Calculate total height
            total_height = 0
            for line in lines:
                bbox = font.getbbox(line)
                total_height += bbox[3] - bbox[1] + 5  # 5px line spacing
            
            if total_height <= max_height:
                return font_size
            
            font_size -= 4
        
        return min_size
    
    def _draw_text_with_outline(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        style: TextStyle
    ) -> None:
        """Draw text with outline/stroke effect."""
        x, y = position
        
        # Draw outline
        for dx in range(-style.stroke_width, style.stroke_width + 1):
            for dy in range(-style.stroke_width, style.stroke_width + 1):
                if dx != 0 or dy != 0:
                    draw.text(
                        (x + dx, y + dy),
                        text,
                        font=font,
                        fill=style.stroke_color
                    )
        
        # Draw main text
        draw.text(position, text, font=font, fill=style.font_color)
    
    def _add_text_to_image(
        self,
        image: Image.Image,
        text: str,
        style: TextStyle,
        position: str = "bottom"
    ) -> Image.Image:
        """Add text to an image at specified position."""
        if not text:
            return image
        
        # Create a copy to avoid modifying original
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Process text
        if style.uppercase:
            text = text.upper()
        
        # Calculate constraints
        img_width, img_height = img.size
        max_width = int(img_width * style.max_width_ratio)
        max_height = int(img_height * 0.25)  # Max 25% of image height per text block
        
        # Get optimal font size
        font_size = self._calculate_font_size(text, style, max_width, max_height)
        font = self._get_font(style, font_size)
        
        # Wrap text
        lines = self._wrap_text(text, font, max_width)
        
        # Calculate total text block height
        line_heights = []
        for line in lines:
            bbox = font.getbbox(line)
            line_heights.append(bbox[3] - bbox[1])
        total_height = sum(line_heights) + (len(lines) - 1) * 5  # 5px spacing
        
        # Calculate starting Y position
        if position == "top":
            y = style.padding
        elif position == "center":
            y = (img_height - total_height) // 2
        else:  # bottom
            y = img_height - total_height - style.padding
        
        # Draw each line
        for i, line in enumerate(lines):
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            x = (img_width - line_width) // 2  # Center horizontally
            
            self._draw_text_with_outline(draw, (x, y), line, font, style)
            y += line_heights[i] + 5
        
        return img
    
    def compose(
        self,
        template: Union[Image.Image, Path, str],
        top_text: Optional[str] = None,
        bottom_text: Optional[str] = None,
        sound_name: Optional[str] = None
    ) -> ComposedMeme:
        """
        Compose a meme from template and text.
        
        Args:
            template: PIL Image, path to image, or Template object
            top_text: Text for top of meme
            bottom_text: Text for bottom of meme
            sound_name: Name of associated sound (for metadata)
        
        Returns:
            ComposedMeme object with the final image
        """
        # Load template if needed
        if isinstance(template, (str, Path)):
            template_path = Path(template)
            template_name = template_path.stem
            img = Image.open(template_path)
        elif hasattr(template, 'load_image'):  # Template object
            template_name = template.name
            img = template.load_image()
        else:
            template_name = "custom"
            img = template
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large
        if img.size[0] > self.config.max_output_size[0] or img.size[1] > self.config.max_output_size[1]:
            img.thumbnail(self.config.max_output_size, Image.Resampling.LANCZOS)
        
        # Add text overlays
        if top_text:
            img = self._add_text_to_image(img, top_text, self.config.top_text_style, "top")
        
        if bottom_text:
            img = self._add_text_to_image(img, bottom_text, self.config.bottom_text_style, "bottom")
        
        return ComposedMeme(
            image=img,
            template_name=template_name,
            top_text=top_text,
            bottom_text=bottom_text,
            sound_name=sound_name
        )
    
    def compose_random(
        self,
        template: Union[Image.Image, Path, str],
        use_top_text: bool = True,
        use_bottom_text: bool = True,
        sound_name: Optional[str] = None
    ) -> ComposedMeme:
        """
        Compose a meme with random caption(s).
        
        Args:
            template: Template to use
            use_top_text: Whether to add top text
            use_bottom_text: Whether to add bottom text
            sound_name: Associated sound name
        
        Returns:
            ComposedMeme with random captions
        """
        top_text = random.choice(self.DEFAULT_CAPTIONS) if use_top_text else None
        bottom_text = random.choice(self.DEFAULT_CAPTIONS) if use_bottom_text else None
        
        return self.compose(template, top_text, bottom_text, sound_name)
    
    @staticmethod
    def get_default_captions() -> List[str]:
        """Get list of default caption options."""
        return MemeComposer.DEFAULT_CAPTIONS.copy()
