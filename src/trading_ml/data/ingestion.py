from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class MarketDataIngestor:
    """Download and store raw market-data files."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, filename: str) -> Path:
        """Download a file into the raw-data directory."""

        destination = self.raw_dir / filename

        if destination.exists():
            logger.info(
                "File already exists; skipping download: %s",
                destination,
            )
            return destination

        logger.info("Downloading: %s", url)

        with urlopen(url) as response:
            data = response.read()

        destination.write_bytes(data)

        logger.info(
            "Downloaded %d bytes to %s",
            len(data),
            destination,
        )

        return destination