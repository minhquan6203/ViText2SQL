"""Evaluate Text-to-SQL predictions using Spider-style EM and component F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.spider_eval.evaluation_core import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá dự đoán SQL theo Exact Match và Component F1")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="File JSON chứa danh sách record có trường 'gold_sql' và 'predict_sql'",
    )
    parser.add_argument(
        "--tables",
        type=str,
        required=True,
        help="File tables.json của database",
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        default=None,
        help="(Tuỳ chọn) Lưu JSON metrics ra file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In chi tiết metrics ra stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_predictions(args.predictions, args.tables)

    if args.verbose:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output_metrics:
        output_path = Path(args.output_metrics)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
