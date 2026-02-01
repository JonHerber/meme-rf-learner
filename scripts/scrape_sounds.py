#!/usr/bin/env python3
"""
CLI script for scraping sounds from myinstants.com.

Usage:
    python scripts/scrape_sounds.py --max-sounds 100
    python scripts/scrape_sounds.py --max-pages 5 --no-download
    python scripts/scrape_sounds.py --resume data/sounds/sounds.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.scrapers.sound_scraper import MyInstantsScraper
from src.scrapers.download_manager import SoundDownloadManager


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
        description="Scrape sound effects from myinstants.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape 50 sounds (quick test)
  python scripts/scrape_sounds.py --max-sounds 50

  # Scrape 100 sounds to custom directory
  python scripts/scrape_sounds.py --max-sounds 100 --output-dir ./my_sounds

  # Scrape without downloading (just collect URLs)
  python scripts/scrape_sounds.py --max-sounds 100 --no-download

  # Resume from previous scrape (download only)
  python scripts/scrape_sounds.py --resume data/sounds/sounds.json

  # Show browser window for debugging
  python scripts/scrape_sounds.py --max-sounds 10 --visible
        """
    )

    parser.add_argument(
        "--max-sounds",
        type=int,
        default=100,
        help="Maximum number of sounds to scrape (default: 100)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to scrape (default: unlimited)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sounds",
        help="Directory to save sounds (default: data/sounds)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Scrape URLs only, don't download files"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous sounds.json file (skip scraping, download only)"
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="(Deprecated - no longer uses browser)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent download workers (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between page requests in seconds (default: 1.5)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sounds.json"

    # Either resume from file or scrape fresh
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        sounds = MyInstantsScraper.load_from_json(args.resume)
        logger.info(f"Loaded {len(sounds)} sounds from file")
    else:
        # Scrape sounds
        logger.info("Starting scraper...")
        logger.info(f"  Max sounds: {args.max_sounds}")
        logger.info(f"  Max pages: {args.max_pages or 'unlimited'}")
        logger.info(f"  Page delay: {args.delay}s")

        scraper = MyInstantsScraper(page_delay=args.delay)

        try:
            sounds = scraper.scrape(
                max_pages=args.max_pages,
                max_sounds=args.max_sounds,
                resolve_mp3_urls=True
            )
        except KeyboardInterrupt:
            logger.warning("Scraping interrupted by user")
            sounds = scraper.sounds
        finally:
            scraper.close()

        # Save scraped data
        if sounds:
            scraper.sounds = sounds
            scraper.save_to_json(str(json_path))
            logger.info(f"Saved {len(sounds)} sounds to {json_path}")
        else:
            logger.error("No sounds scraped")
            return 1

    # Download sounds
    if not args.no_download:
        sounds_with_urls = [s for s in sounds if s.mp3_url]
        logger.info(f"Downloading {len(sounds_with_urls)} sounds...")

        downloader = SoundDownloadManager(
            output_dir=str(output_dir),
            max_workers=args.workers
        )

        stats = downloader.download_all(sounds, skip_existing=True)

        logger.info("=" * 50)
        logger.info("Download Summary:")
        logger.info(f"  Total sounds: {stats['total']}")
        logger.info(f"  Downloaded: {stats['success']}")
        logger.info(f"  Skipped (existing): {stats['skipped']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  No URL: {stats['no_url']}")
        logger.info(f"  Output directory: {output_dir.absolute()}")
        logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
