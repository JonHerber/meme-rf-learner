#!/usr/bin/env python3
"""
CLI script for downloading meme templates from Google Drive.

Usage:
    python scripts/download_templates.py
    python scripts/download_templates.py --url "https://drive.google.com/drive/folders/..."
    python scripts/download_templates.py --output-dir ./my_templates
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data.drive_loader import DriveLoader
from src.data.template_manager import TemplateManager


# Default Google Drive folder URL
DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/1UXKquhbrh_aC48FeqY60TW6YXls9gAMD"


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download meme templates from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from default folder
  python scripts/download_templates.py

  # Download from custom folder
  python scripts/download_templates.py --url "https://drive.google.com/drive/folders/YOUR_FOLDER_ID"

  # Download to custom directory
  python scripts/download_templates.py --output-dir ./my_templates

  # Force re-download (skip nothing)
  python scripts/download_templates.py --no-skip
        """
    )

    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_FOLDER_URL,
        help="Google Drive folder URL or ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/templates",
        help="Directory to save templates (default: data/templates)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download all files (don't skip existing)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress gdown progress output"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("Meme Template Downloader")
    logger.info("=" * 50)

    # Initialize loader
    loader = DriveLoader(
        output_dir=args.output_dir,
        quiet=args.quiet
    )

    # Download templates
    try:
        stats = loader.download_folder(
            folder_url_or_id=args.url,
            skip_existing=not args.no_skip
        )
    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        return 1
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1

    # Save metadata
    if loader.templates:
        loader.save_metadata()

    # Verify with TemplateManager
    manager = TemplateManager(template_dir=args.output_dir)
    template_count = len(manager)

    # Summary
    logger.info("=" * 50)
    logger.info("Download Summary:")
    logger.info(f"  Total files processed: {stats['total']}")
    logger.info(f"  Downloaded: {stats['downloaded']}")
    logger.info(f"  Skipped (existing): {stats['skipped']}")
    logger.info(f"  Non-image files: {stats['non_image']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Templates available: {template_count}")
    logger.info(f"  Output directory: {Path(args.output_dir).absolute()}")
    logger.info("=" * 50)

    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
