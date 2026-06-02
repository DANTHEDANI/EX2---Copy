import argparse

from .sdk import TradingSDK


def main() -> None:
    """Main CLI entrypoint routing through TradingSDK only."""
    parser = argparse.ArgumentParser(description="Educational Dueling-DQN trading CLI")
    parser.add_argument(
        "--action", type=str, choices=["train", "backtest"], default="train", help="SDK action."
    )
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker to process")
    args = parser.parse_args()
    sdk = TradingSDK()
    if args.action == "train":
        sdk.run_training_pipeline(args.ticker)
    else:
        sdk.evaluate_strategy(args.ticker)

if __name__ == "__main__":
    main()
