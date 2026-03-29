"""Build the processed dataset and run explicit leakage audits."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_aligned_market_data
from src.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PROCESSED_DIR,
    build_feature_frame,
    leakage_audit,
    save_processed_dataset,
    split_by_date_range,
)


def main() -> None:
    raw = load_aligned_market_data(force=False)
    feature_frame = build_feature_frame(raw)
    outputs = save_processed_dataset(feature_frame, output_dir=PROCESSED_DIR)

    splits = split_by_date_range(feature_frame)

    print("Processed dataset built successfully")
    for name, subset in splits.items():
        print(f"{name}: {len(subset)} rows")

    final_df = feature_frame[FEATURE_COLUMNS + [TARGET_COLUMN]]
    print("NaN counts in final usable dataset:")
    print(final_df.isna().sum().to_string())

    audit_df, passed = leakage_audit(feature_frame, raw)
    print("Leakage audit:")
    print(audit_df.to_string(index=False))

    if not passed:
        raise SystemExit("Leakage audit failed. Stop and inspect the feature construction logic.")

    print("Leakage audit passed")
    print("Saved files:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()

