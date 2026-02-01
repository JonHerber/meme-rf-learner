"""
Scraper for myinstants.com sound effects.

Uses requests + BeautifulSoup for efficient scraping without browser dependencies.
"""

import time
import re
from typing import List, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class SoundInfo:
    """Information about a scraped sound."""
    name: str
    detail_url: str
    mp3_url: Optional[str] = None
    downloaded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class MyInstantsScraper:
    """
    Scrapes sounds from myinstants.com.

    Uses requests + BeautifulSoup for fast, lightweight scraping.
    """

    BASE_URL = "https://www.myinstants.com"
    INDEX_URL = f"{BASE_URL}/en/index/us/"

    def __init__(
        self,
        page_delay: float = 1.0,
        timeout: int = 15,
        headless: bool = True  # Kept for API compatibility, not used
    ):
        """
        Initialize the scraper.

        Args:
            page_delay: Delay between page requests (seconds)
            timeout: Request timeout (seconds)
            headless: Ignored (kept for API compatibility)
        """
        self.page_delay = page_delay
        self.timeout = timeout
        self.session = self._create_session()
        self.sounds: List[SoundInfo] = []

    def _create_session(self) -> requests.Session:
        """Create a requests session with appropriate headers."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        return session

    def start(self):
        """Initialize session (for API compatibility)."""
        logger.info("Scraper initialized")

    def close(self):
        """Close the session."""
        self.session.close()
        logger.info("Scraper closed")

    def scrape_listing_page(self, page: int = 1) -> List[SoundInfo]:
        """
        Scrape a single listing page for sound info.

        The MP3 URLs are directly available in the button onclick attributes,
        so we can extract everything from the listing page.

        Args:
            page: Page number to scrape

        Returns:
            List of SoundInfo objects from that page
        """
        url = f"{self.INDEX_URL}?page={page}"
        logger.info(f"Scraping page {page}: {url}")

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch page {page}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        sounds = []

        # Find all instant div elements
        instant_elements = soup.find_all("div", class_="instant")

        if not instant_elements:
            logger.info(f"No sounds found on page {page}")
            return []

        for elem in instant_elements:
            try:
                # Find the link with sound name
                link = elem.find("a", class_="instant-link")
                if not link:
                    link = elem.find("a", href=lambda x: x and "/instant/" in x)

                if not link:
                    continue

                name = link.get_text(strip=True)
                href = link.get("href", "")

                # Make URL absolute
                if href and not href.startswith("http"):
                    detail_url = f"{self.BASE_URL}{href}"
                else:
                    detail_url = href

                # Extract MP3 URL from button onclick
                mp3_url = None
                button = elem.find("button", class_="small-button")
                if button:
                    onclick = button.get("onclick", "")
                    # Pattern: play('/media/sounds/filename.mp3', ...)
                    match = re.search(r"play\s*\(\s*['\"]([^'\"]+\.mp3)['\"]", onclick)
                    if match:
                        mp3_path = match.group(1)
                        mp3_url = f"{self.BASE_URL}{mp3_path}" if not mp3_path.startswith("http") else mp3_path

                if name and detail_url:
                    sound = SoundInfo(
                        name=name,
                        detail_url=detail_url,
                        mp3_url=mp3_url
                    )
                    sounds.append(sound)

            except Exception as e:
                logger.debug(f"Error parsing sound element: {e}")
                continue

        logger.info(f"Found {len(sounds)} sounds on page {page}")
        return sounds

    def get_mp3_url(self, sound: SoundInfo) -> Optional[str]:
        """
        Fetch the detail page to extract the MP3 URL.
        Only needed if MP3 URL wasn't found on listing page.

        Args:
            sound: SoundInfo object with detail_url set

        Returns:
            MP3 URL or None if not found
        """
        if sound.mp3_url:
            return sound.mp3_url

        try:
            response = self.session.get(sound.detail_url, timeout=self.timeout)
            response.raise_for_status()
            html = response.text

            # Method 1: Look for preloadAudioUrl JavaScript variable
            match = re.search(r"preloadAudioUrl\s*=\s*['\"]([^'\"]+\.mp3)['\"]", html)
            if match:
                mp3_path = match.group(1)
                return f"{self.BASE_URL}{mp3_path}" if not mp3_path.startswith("http") else mp3_path

            # Method 2: Look for play() function call
            match = re.search(r"play\s*\(\s*['\"]([^'\"]+\.mp3)['\"]", html)
            if match:
                mp3_path = match.group(1)
                return f"{self.BASE_URL}{mp3_path}" if not mp3_path.startswith("http") else mp3_path

            # Method 3: Parse HTML for download link
            soup = BeautifulSoup(html, "html.parser")
            download_link = soup.find("a", href=lambda x: x and x.endswith(".mp3"))
            if download_link:
                href = download_link.get("href", "")
                return f"{self.BASE_URL}{href}" if not href.startswith("http") else href

            logger.warning(f"Could not find MP3 URL for: {sound.name}")
            return None

        except requests.RequestException as e:
            logger.error(f"Error fetching detail page for {sound.name}: {e}")
            return None

    def scrape(
        self,
        max_pages: Optional[int] = None,
        max_sounds: Optional[int] = None,
        resolve_mp3_urls: bool = True
    ) -> List[SoundInfo]:
        """
        Scrape sounds from myinstants.com.

        Args:
            max_pages: Maximum pages to scrape (None for unlimited)
            max_sounds: Maximum sounds to collect (None for unlimited)
            resolve_mp3_urls: Whether to fetch detail pages for missing MP3 URLs

        Returns:
            List of SoundInfo objects
        """
        self.start()
        self.sounds = []

        page = 1
        consecutive_empty = 0

        try:
            while True:
                # Check limits
                if max_pages and page > max_pages:
                    logger.info(f"Reached max pages limit: {max_pages}")
                    break

                if max_sounds and len(self.sounds) >= max_sounds:
                    logger.info(f"Reached max sounds limit: {max_sounds}")
                    self.sounds = self.sounds[:max_sounds]
                    break

                # Scrape page
                page_sounds = self.scrape_listing_page(page)

                if not page_sounds:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        logger.info("3 consecutive empty pages, stopping")
                        break
                else:
                    consecutive_empty = 0
                    self.sounds.extend(page_sounds)

                page += 1
                time.sleep(self.page_delay)

            # Resolve missing MP3 URLs if requested
            if resolve_mp3_urls:
                missing = [s for s in self.sounds if not s.mp3_url]
                if missing:
                    logger.info(f"Resolving {len(missing)} missing MP3 URLs...")
                    for i, sound in enumerate(missing):
                        mp3_url = self.get_mp3_url(sound)
                        sound.mp3_url = mp3_url
                        if (i + 1) % 10 == 0:
                            logger.info(f"Resolved {i + 1}/{len(missing)} MP3 URLs")
                        time.sleep(0.3)

        finally:
            self.close()

        # Count successful resolutions
        resolved = sum(1 for s in self.sounds if s.mp3_url)
        logger.info(f"Scraping complete. Found {len(self.sounds)} sounds, {resolved} with MP3 URLs")

        return self.sounds

    def save_to_json(self, filepath: str):
        """Save scraped sounds to JSON file."""
        data = [s.to_dict() for s in self.sounds]
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.sounds)} sounds to {filepath}")

    @classmethod
    def load_from_json(cls, filepath: str) -> List[SoundInfo]:
        """Load sounds from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [SoundInfo(**item) for item in data]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
