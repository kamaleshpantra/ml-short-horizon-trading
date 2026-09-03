from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class MarketDataIngestor:
    """Download and manage raw market-data files."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        url: str,
        filename: str,
        expected_sha256: str | None = None,
    ) -> Path:
        """
        Download a file and optionally verify its SHA-256 checksum.

        Parameters
        ----------
        url:
            Source URL.

        filename:
            Destination filename inside raw_dir.

        expected_sha256:
            Optional expected SHA-256 checksum.
        """

        destination = self.raw_dir / filename

        if destination.exists():
            logger.info(
                "File already exists: %s",
                destination,
            )

            if expected_sha256:
                self._verify_checksum(
                    destination,
                    expected_sha256,
                )

            return destination

        logger.info("Starting download: %s", url)

        try:
            with urlopen(url) as response:
                data = response.read()

        except Exception:
            logger.exception(
                "Failed to download: %s",
                url,
            )
            raise

        destination.write_bytes(data)

        logger.info(
            "Downloaded %.2f MB to %s",
            len(data) / (1024 * 1024),
            destination,
        )

        if expected_sha256:
            self._verify_checksum(
                destination,
                expected_sha256,
            )

        return destination

    @staticmethod
    def _calculate_sha256(path: Path) -> str:
        """Calculate SHA-256 checksum for a file."""

        sha256 = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _verify_checksum(
        self,
        path: Path,
        expected_sha256: str,
    ) -> None:
        """Verify file integrity."""

        actual_sha256 = self._calculate_sha256(path)

        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {path}. "
                f"Expected {expected_sha256}, "
                f"got {actual_sha256}."
            )

        logger.info(
            "Checksum verified: %s",
            path,
        )