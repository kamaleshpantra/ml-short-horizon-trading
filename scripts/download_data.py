from trading_ml.utils.config import load_config
from trading_ml.utils.logging import configure_logging


def main() -> None:
    configure_logging()

    config = load_config("configs/config.yaml")

    symbol = config["data"]["symbol"]

    print(f"Configured symbol: {symbol}")


if __name__ == "__main__":
    main()