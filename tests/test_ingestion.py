from pathlib import Path

from trading_ml.data.ingestion import MarketDataIngestor


def test_ingestor_creates_raw_directory(tmp_path: Path):

    raw_dir = tmp_path / "raw"

    ingestor = MarketDataIngestor(raw_dir)

    assert raw_dir.exists()