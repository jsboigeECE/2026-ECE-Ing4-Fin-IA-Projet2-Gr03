"""Download and cache SPY and VIX daily data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_or_download_spy, load_or_download_vix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache SPY/VIX daily data.")
    parser.add_argument("--force", action="store_true", help="Force a fresh download even if cache files exist.")
    parser.add_argument("--start", default=None, help="Optional yfinance start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Optional yfinance end date (YYYY-MM-DD, exclusive).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = args.start or "2003-12-01"
    end = args.end or "2025-01-01"
    spy = load_or_download_spy(force=args.force, start=start, end=end)
    vix = load_or_download_vix(force=args.force, start=start, end=end)

    print("Raw data cached successfully")
    print(f"SPY rows: {len(spy)} | date range: {spy.index.min().date()} -> {spy.index.max().date()}")
    print(f"VIX rows: {len(vix)} | date range: {vix.index.min().date()} -> {vix.index.max().date()}")


if __name__ == "__main__":
    main()

