from pathlib import Path

import pytest

from trading_ml.data.ingestion import MarketDataIngestor


def test_ingestor_creates_raw_directory(tmp_path: Path):

    raw_dir = tmp_path / "raw"

    ingestor = MarketDataIngestor(raw_dir)

    assert raw_dir.exists()


def test_download_local_file(tmp_path: Path):

    source = tmp_path / "source.txt"
    source.write_text(
        "market data",
        encoding="utf-8",
    )

    raw_dir = tmp_path / "raw"

    ingestor = MarketDataIngestor(raw_dir)

    destination = ingestor.download(
        source.as_uri(),
        "market.txt",
    )

    assert destination.exists()

    assert destination.read_text(
        encoding="utf-8"
    ) == "market data"


def test_checksum_validation(tmp_path: Path):

    source = tmp_path / "source.txt"

    source.write_text(
        "market data",
        encoding="utf-8",
    )

    raw_dir = tmp_path / "raw"

    ingestor = MarketDataIngestor(raw_dir)

    with pytest.raises(ValueError, match="Checksum mismatch"):

        ingestor.download(
            source.as_uri(),
            "market.txt",
            expected_sha256="incorrect_checksum",
        )