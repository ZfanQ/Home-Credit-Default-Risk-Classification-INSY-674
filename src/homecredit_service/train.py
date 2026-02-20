from __future__ import annotations

import argparse
import json
from pathlib import Path

from homecredit_service.features import build_training_frame
from homecredit_service.modeling import save_bundle, train_lightgbm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Home Credit default risk LightGBM model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("homecreditdefaultriskdata"),
        help="Directory containing all raw Home Credit CSV files.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for model artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row count for faster experimentation.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_frame = build_training_frame(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    bundle = train_lightgbm_model(
        train_df=training_frame,
        random_state=args.random_state,
        valid_size=args.valid_size,
    )

    artifact_path = args.artifact_dir / "model_bundle.joblib"
    save_bundle(bundle, artifact_path)

    report = {
        "artifact_path": str(artifact_path),
        "validation_auc": bundle["metrics"]["validation_auc"],
        "scale_pos_weight": bundle["scale_pos_weight"],
        "feature_count": len(bundle["feature_columns"]),
        "train_rows": bundle["train_rows"],
        "valid_rows": bundle["valid_rows"],
        "trained_at_utc": bundle["trained_at_utc"],
    }

    report_path = args.artifact_dir / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
