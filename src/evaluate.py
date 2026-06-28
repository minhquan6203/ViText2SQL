"""
Script đánh giá Exact Match (EM) và Component F1 cho Text-to-SQL.

Logic đánh giá kế thừa từ Spider Benchmark (evaluation.py):
- EM: so khớp cấu trúc AST SQL (không phải so sánh chuỗi)
- Component F1: Precision/Recall/F1 cho SELECT, WHERE, GROUP BY, ORDER BY

Tham khảo: https://github.com/taoyds/spider/blob/master/evaluation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.spider_eval.evaluation_core import evaluate_batch, evaluate_em_f1
from src.spider_eval.process_sql import build_schema_from_tables_entry, Schema
from src.utils import load_json, save_json


def predictions_to_spider_format(
    predictions: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Chuyển prediction.json sang định dạng Spider (mỗi dòng: SQL\\tdb_id).

    Dùng cho tương thích với script evaluate.py gốc của Spider.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for item in predictions:
            sql = item.get("predict_sql") or item.get("predicted_sql", "")
            db_id = item["db_id"]
            file.write(f"{sql}\t{db_id}\n")


def gold_to_spider_format(
    predictions: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Ghi file gold SQL theo định dạng Spider."""
    with open(output_path, "w", encoding="utf-8") as file:
        for item in predictions:
            sql = item.get("gold_sql") or item.get("query", "")
            db_id = item["db_id"]
            file.write(f"{sql}\t{db_id}\n")


def print_evaluation_report(metrics: Dict[str, float], title: str = "Kết quả đánh giá") -> None:
    """In báo cáo đánh giá ra console."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Số mẫu:              {metrics.get('total_samples', 0)}")
    print(f"  Exact Match (EM):    {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.2f}%)")
    print("-" * 60)
    print("  Component F1 Scores:")
    print(f"    SELECT F1:         {metrics['select_f1']:.4f}")
    print(f"    WHERE F1:          {metrics['where_f1']:.4f}")
    print(f"    GROUP BY F1:       {metrics['group_f1']:.4f}")
    print(f"    ORDER BY F1:       {metrics['order_f1']:.4f}")
    print(f"    Trung bình F1:     {metrics['average_component_f1']:.4f}")
    print("=" * 60 + "\n")


def evaluate_predictions_file(
    predictions_path: Path,
    tables_path: Path,
    output_metrics_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Đánh giá file prediction.json và trả về metrics.

    Args:
        predictions_path: Đường dẫn prediction.json
        tables_path: Đường dẫn tables.json
        output_metrics_path: Lưu metrics ra JSON (tùy chọn)
        verbose: In chi tiết từng mẫu sai
    """
    predictions = load_json(predictions_path)
    tables_data = load_json(tables_path)
    tables_by_id = {entry["db_id"]: entry for entry in tables_data}

    if verbose:
        for index, item in enumerate(predictions):
            db_id = item["db_id"]
            table_entry = tables_by_id[db_id]
            schema = Schema(build_schema_from_tables_entry(table_entry))

            result = evaluate_em_f1(
                predict_sql=item.get("predict_sql", ""),
                gold_sql=item.get("gold_sql", ""),
                db_schema=schema,
                gold_tokens=item.get("gold_query_toks"),
            )

            if result["exact_match"] == 0:
                print(f"\n[Mẫu {index}] db_id={db_id}")
                print(f"  Câu hỏi: {item.get('question', '')[:80]}...")
                print(f"  Gold:    {item.get('gold_sql', '')[:120]}...")
                print(f"  Pred:    {item.get('predict_sql', '')[:120]}...")

    metrics = evaluate_batch(predictions, tables_path=str(tables_path), tables_data=tables_data)
    print_evaluation_report(metrics, title=f"Đánh giá: {predictions_path.name}")

    if output_metrics_path:
        save_json(metrics, output_metrics_path)

    return metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá EM và Component F1 cho Text-to-SQL (chuẩn Spider/ViText2SQL)"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Đường dẫn file prediction.json",
    )
    parser.add_argument(
        "--tables",
        type=str,
        default="data/word-level/tables.json",
        help="Đường dẫn tables.json",
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        default=None,
        help="Lưu metrics ra file JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In chi tiết các mẫu sai",
    )
    parser.add_argument(
        "--export_spider_format",
        type=str,
        default=None,
        help="Xuất pred/gold theo định dạng Spider (.sql tab-separated)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    predictions_path = Path(args.predictions)
    tables_path = Path(args.tables)

    if args.export_spider_format:
        export_dir = Path(args.export_spider_format)
        export_dir.mkdir(parents=True, exist_ok=True)
        predictions = load_json(predictions_path)
        predictions_to_spider_format(predictions, export_dir / "predict.sql")
        gold_to_spider_format(predictions, export_dir / "gold.sql")
        print(f"Đã xuất định dạng Spider tại: {export_dir}")

    output_metrics = Path(args.output_metrics) if args.output_metrics else None
    evaluate_predictions_file(
        predictions_path=predictions_path,
        tables_path=tables_path,
        output_metrics_path=output_metrics,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
