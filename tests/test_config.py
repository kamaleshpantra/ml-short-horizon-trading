from trading_ml.utils.config import load_config


def test_load_config():

    config = load_config("configs/config.yaml")

    assert config["project"]["name"] == "ml-short-horizon-trading"
    assert config["data"]["symbol"] == "BTCUSDT"
    assert config["data"]["orderbook_levels"] == 5