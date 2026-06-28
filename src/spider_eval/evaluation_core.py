"""
Logic đánh giá Exact Match (EM) và Component F1 kế thừa từ Spider Benchmark.

Tham khảo: https://github.com/taoyds/spider/blob/master/evaluation.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from src.spider_eval.process_sql import (
    Schema,
    build_schema_from_tables_entry,
    build_schema_map_from_tables_json,
    empty_sql,
    get_sql,
)

DISABLE_VALUE = True
DISABLE_DISTINCT = True

WHERE_OPS = (
    "not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists",
)
TABLE_TYPE = {"sql": "sql", "table_unit": "table_unit"}


def get_scores(count: int, pred_total: int, label_total: int) -> Tuple[int, int, int]:
    """Tính accuracy, recall, F1 nhị phân cho một thành phần."""
    if pred_total != label_total:
        return 0, 0, 0
    if count == pred_total:
        return 1, 1, 1
    return 0, 0, 0


def eval_sel(pred: Dict, label: Dict) -> Tuple[int, int, int, int]:
    pred_sel = pred["select"][1]
    label_sel = label["select"][1]
    label_without_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    count = 0
    count_without_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            count += 1
            label_sel.remove(unit)
        if unit[1] in label_without_agg:
            count_without_agg += 1
            label_without_agg.remove(unit[1])

    return label_total, pred_total, count, count_without_agg


def eval_where(pred: Dict, label: Dict) -> Tuple[int, int, int, int]:
    pred_conds = [unit for unit in pred["where"][::2]]
    label_conds = [unit for unit in label["where"][::2]]
    label_without_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    count = 0
    count_without_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            count += 1
            label_conds.remove(unit)
        if unit[2] in label_without_agg:
            count_without_agg += 1
            label_without_agg.remove(unit[2])

    return label_total, pred_total, count, count_without_agg


def eval_group(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    pred_cols = [unit[1] for unit in pred["groupBy"]]
    label_cols = [unit[1] for unit in label["groupBy"]]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    count = 0
    pred_cols = [col.split(".")[1] if "." in col else col for col in pred_cols]
    label_cols = [col.split(".")[1] if "." in col else col for col in label_cols]
    for col in pred_cols:
        if col in label_cols:
            count += 1
            label_cols.remove(col)
    return label_total, pred_total, count


def eval_having(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    pred_total = label_total = count = 0
    if len(pred["groupBy"]) > 0:
        pred_total = 1
    if len(label["groupBy"]) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred["groupBy"]]
    label_cols = [unit[1] for unit in label["groupBy"]]
    if (
        pred_total == label_total == 1
        and pred_cols == label_cols
        and pred["having"] == label["having"]
    ):
        count = 1
    return label_total, pred_total, count


def eval_order(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    pred_total = label_total = count = 0
    if len(pred["orderBy"]) > 0:
        pred_total = 1
    if len(label["orderBy"]) > 0:
        label_total = 1
    if (
        len(label["orderBy"]) > 0
        and pred["orderBy"] == label["orderBy"]
        and (
            (pred["limit"] is None and label["limit"] is None)
            or (pred["limit"] is not None and label["limit"] is not None)
        )
    ):
        count = 1
    return label_total, pred_total, count


def eval_and_or(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    pred_ao = set(pred["where"][1::2])
    label_ao = set(label["where"][1::2])
    if pred_ao == label_ao:
        return 1, 1, 1
    return len(pred_ao), len(label_ao), 0


def eval_nested(pred: Optional[Dict], label: Optional[Dict], evaluator: "Evaluator") -> Tuple[int, int, int]:
    label_total = pred_total = count = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        count += evaluator.eval_exact_match(pred, label)
    return label_total, pred_total, count


def eval_iuen(pred: Dict, label: Dict, evaluator: "Evaluator") -> Tuple[int, int, int]:
    lt1, pt1, cnt1 = eval_nested(pred["intersect"], label["intersect"], evaluator)
    lt2, pt2, cnt2 = eval_nested(pred["except"], label["except"], evaluator)
    lt3, pt3, cnt3 = eval_nested(pred["union"], label["union"], evaluator)
    return lt1 + lt2 + lt3, pt1 + pt2 + pt3, cnt1 + cnt2 + cnt3


def get_keywords(sql: Dict) -> set:
    result = set()
    if len(sql["where"]) > 0:
        result.add("where")
    if len(sql["groupBy"]) > 0:
        result.add("group")
    if len(sql["having"]) > 0:
        result.add("having")
    if len(sql["orderBy"]) > 0:
        result.add(sql["orderBy"][0])
        result.add("order")
    if sql["limit"] is not None:
        result.add("limit")
    if sql["except"] is not None:
        result.add("except")
    if sql["union"] is not None:
        result.add("union")
    if sql["intersect"] is not None:
        result.add("intersect")

    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    if any(token == "or" for token in ao):
        result.add("or")

    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    if any(cond_unit[0] for cond_unit in cond_units):
        result.add("not")
    if any(cond_unit[1] == WHERE_OPS.index("in") for cond_unit in cond_units):
        result.add("in")
    if any(cond_unit[1] == WHERE_OPS.index("like") for cond_unit in cond_units):
        result.add("like")
    return result


def eval_keywords(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    count = sum(1 for keyword in pred_keywords if keyword in label_keywords)
    return label_total, pred_total, count


def build_foreign_key_map(entry: Dict[str, Any]) -> Dict[str, str]:
    """Xây dựng bản đồ khóa ngoại từ mục tables.json."""
    cols_original = entry["column_names_original"]
    tables_original = entry["table_names_original"]

    cols: List[str] = []
    for col_orig in cols_original:
        if col_orig[0] >= 0:
            table = tables_original[col_orig[0]]
            column = col_orig[1]
            cols.append(f"__{table.lower()}.{column.lower()}__")
        else:
            cols.append("__all__")

    def keyset_in_list(key1: int, key2: int, key_list: List[set]) -> set:
        for key_set in key_list:
            if key1 in key_set or key2 in key_set:
                return key_set
        new_key_set: set = set()
        key_list.append(new_key_set)
        return new_key_set

    foreign_key_list: List[set] = []
    for foreign_key in entry["foreign_keys"]:
        key1, key2 = foreign_key
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map: Dict[str, str] = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        master_index = sorted_list[0]
        for index in sorted_list:
            foreign_key_map[cols[index]] = cols[master_index]
    return foreign_key_map


def build_foreign_key_map_from_json(tables_path: str) -> Dict[str, Dict[str, str]]:
    with open(tables_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {entry["db_id"]: build_foreign_key_map(entry) for entry in data}


def build_valid_col_units(table_units: List, schema: Schema) -> List[str]:
    col_ids = [
        table_unit[1]
        for table_unit in table_units
        if table_unit[0] == TABLE_TYPE["table_unit"]
    ]
    prefixes = [col_id[:-2] for col_id in col_ids]
    valid_col_units: List[str] = []
    for value in schema.idMap.values():
        if "." in value and value[: value.index(".")] in prefixes:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units: List[str], col_unit: Any, key_map: Dict[str, str]) -> Any:
    if col_unit is None:
        return col_unit
    agg_id, col_id, distinct = col_unit
    if col_id in key_map and col_id in valid_col_units:
        col_id = key_map[col_id]
    if DISABLE_DISTINCT:
        distinct = None
    return agg_id, col_id, distinct


def rebuild_val_unit_col(valid_col_units: List[str], val_unit: Any, key_map: Dict[str, str]) -> Any:
    if val_unit is None:
        return val_unit
    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, key_map)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, key_map)
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units: List[str], table_unit: Any, key_map: Dict[str, str]) -> Any:
    if table_unit is None:
        return table_unit
    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, key_map)
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units: List[str], cond_unit: Any, key_map: Dict[str, str]) -> Any:
    if cond_unit is None:
        return cond_unit
    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, key_map)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_col(valid_col_units: List[str], condition: List, key_map: Dict[str, str]) -> List:
    for index in range(len(condition)):
        if index % 2 == 0:
            condition[index] = rebuild_cond_unit_col(valid_col_units, condition[index], key_map)
    return condition


def rebuild_select_col(valid_col_units: List[str], select_clause: Any, key_map: Dict[str, str]) -> Any:
    if select_clause is None:
        return select_clause
    distinct, item_list = select_clause
    new_list = []
    for item in item_list:
        agg_id, val_unit = item
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, key_map)))
    if DISABLE_DISTINCT:
        distinct = None
    return distinct, new_list


def rebuild_from_col(valid_col_units: List[str], from_clause: Dict, key_map: Dict[str, str]) -> Dict:
    if from_clause is None:
        return from_clause
    from_clause["table_units"] = [
        rebuild_table_unit_col(valid_col_units, table_unit, key_map)
        for table_unit in from_clause["table_units"]
    ]
    from_clause["conds"] = rebuild_condition_col(valid_col_units, from_clause["conds"], key_map)
    return from_clause


def rebuild_group_by_col(valid_col_units: List[str], group_by: List, key_map: Dict[str, str]) -> List:
    if group_by is None:
        return group_by
    return [rebuild_col_unit_col(valid_col_units, col_unit, key_map) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units: List[str], order_by: Any, key_map: Dict[str, str]) -> Any:
    if order_by is None or len(order_by) == 0:
        return order_by
    direction, val_units = order_by
    new_val_units = [
        rebuild_val_unit_col(valid_col_units, val_unit, key_map) for val_unit in val_units
    ]
    return direction, new_val_units


def rebuild_cond_unit_val(cond_unit: Any) -> Any:
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit
    not_op, op_id, val_unit, val1, val2 = cond_unit
    if not isinstance(val1, dict):
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    if not isinstance(val2, dict):
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_val(condition: List) -> List:
    if condition is None or not DISABLE_VALUE:
        return condition
    result: List[Any] = []
    for index, item in enumerate(condition):
        if index % 2 == 0:
            result.append(rebuild_cond_unit_val(item))
        else:
            result.append(item)
    return result


def rebuild_sql_val(sql: Optional[Dict]) -> Optional[Dict]:
    if sql is None or not DISABLE_VALUE:
        return sql
    sql["from"]["conds"] = rebuild_condition_val(sql["from"]["conds"])
    sql["having"] = rebuild_condition_val(sql["having"])
    sql["where"] = rebuild_condition_val(sql["where"])
    sql["intersect"] = rebuild_sql_val(sql["intersect"])
    sql["except"] = rebuild_sql_val(sql["except"])
    sql["union"] = rebuild_sql_val(sql["union"])
    return sql


def rebuild_sql_col(valid_col_units: List[str], sql: Optional[Dict], key_map: Dict[str, str]) -> Optional[Dict]:
    if sql is None:
        return sql
    sql["select"] = rebuild_select_col(valid_col_units, sql["select"], key_map)
    sql["from"] = rebuild_from_col(valid_col_units, sql["from"], key_map)
    sql["where"] = rebuild_condition_col(valid_col_units, sql["where"], key_map)
    sql["groupBy"] = rebuild_group_by_col(valid_col_units, sql["groupBy"], key_map)
    sql["orderBy"] = rebuild_order_by_col(valid_col_units, sql["orderBy"], key_map)
    sql["having"] = rebuild_condition_col(valid_col_units, sql["having"], key_map)
    sql["intersect"] = rebuild_sql_col(valid_col_units, sql["intersect"], key_map)
    sql["except"] = rebuild_sql_col(valid_col_units, sql["except"], key_map)
    sql["union"] = rebuild_sql_col(valid_col_units, sql["union"], key_map)
    return sql


class Evaluator:
    """Bộ đánh giá Exact Match và Partial Match theo chuẩn Spider."""

    def __init__(self) -> None:
        self.partial_scores: Optional[Dict[str, Dict[str, float]]] = None

    def eval_exact_match(self, pred: Dict, label: Dict) -> int:
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores

        for _, score in partial_scores.items():
            if score["f1"] != 1:
                return 0

        if len(label["from"]["table_units"]) > 0:
            label_tables = sorted(label["from"]["table_units"])
            pred_tables = sorted(pred["from"]["table_units"])
            return 1 if label_tables == pred_tables else 0
        return 1

    def eval_partial_match(self, pred: Dict, label: Dict) -> Dict[str, Dict[str, Union[int, float]]]:
        result: Dict[str, Dict[str, Union[int, float]]] = {}

        label_total, pred_total, count, count_without_agg = eval_sel(pred, label)
        accuracy, recall, f1 = get_scores(count, pred_total, label_total)
        result["select"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }
        accuracy, recall, f1 = get_scores(count_without_agg, pred_total, label_total)
        result["select(no AGG)"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }

        label_total, pred_total, count, count_without_agg = eval_where(pred, label)
        accuracy, recall, f1 = get_scores(count, pred_total, label_total)
        result["where"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }
        accuracy, recall, f1 = get_scores(count_without_agg, pred_total, label_total)
        result["where(no OP)"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }

        label_total, pred_total, count = eval_group(pred, label)
        accuracy, recall, f1 = get_scores(count, pred_total, label_total)
        result["group(no Having)"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }

        label_total, pred_total, count = eval_having(pred, label)
        accuracy, recall, f1 = get_scores(count, pred_total, label_total)
        result["group"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }

        label_total, pred_total, count = eval_order(pred, label)
        accuracy, recall, f1 = get_scores(count, pred_total, label_total)
        result["order"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }

        label_total, pred_total, count = eval_and_or(pred, label)
        accuracy, recall, f1 = get_scores(count, pred_total, label_total)
        result["and/or"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }

        label_total, pred_total, count = eval_iuen(pred, label, self)
        accuracy, recall, f1 = get_scores(count, pred_total, label_total)
        result["IUEN"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }

        label_total, pred_total, count = eval_keywords(pred, label)
        accuracy, recall, f1 = get_scores(count, pred_total, label_total)
        result["keywords"] = {
            "acc": accuracy, "rec": recall, "f1": f1,
            "label_total": label_total, "pred_total": pred_total,
        }

        return result


def evaluate_em_f1(
    predict_sql: str,
    gold_sql: str,
    db_schema: Union[Schema, Dict[str, List[str]], Dict[str, Any]],
    gold_tokens: Optional[List[str]] = None,
    predict_tokens: Optional[List[str]] = None,
    foreign_key_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Đánh giá một cặp (SQL dự đoán, SQL gốc) theo EM và Component F1.

    Args:
        predict_sql: Câu SQL do mô hình sinh ra
        gold_sql: Câu SQL gốc (ground truth)
        db_schema: Schema (Schema object, dict schema, hoặc mục tables.json)
        gold_tokens: query_toks gốc (nếu có, parse chính xác hơn)
        predict_tokens: Token dự đoán (nếu có)
        foreign_key_map: Bản đồ khóa ngoại cho chuẩn hóa cột

    Returns:
        Dictionary chứa exact_match, partial_scores, và f1 trung bình các thành phần chính
    """
    if isinstance(db_schema, Schema):
        schema = db_schema
    elif isinstance(db_schema, dict) and "table_names" in db_schema:
        schema = Schema(build_schema_from_tables_entry(db_schema))
    else:
        schema = Schema(db_schema)

    key_map = foreign_key_map or {}
    evaluator = Evaluator()

    try:
        gold_parsed = get_sql(schema, gold_tokens if gold_tokens else gold_sql)
    except Exception:
        gold_parsed = empty_sql()

    try:
        pred_parsed = get_sql(schema, predict_tokens if predict_tokens else predict_sql)
    except Exception:
        pred_parsed = empty_sql()

    gold_valid_cols = build_valid_col_units(gold_parsed["from"]["table_units"], schema)
    gold_parsed = rebuild_sql_val(gold_parsed)
    gold_parsed = rebuild_sql_col(gold_valid_cols, gold_parsed, key_map)

    pred_valid_cols = build_valid_col_units(pred_parsed["from"]["table_units"], schema)
    pred_parsed = rebuild_sql_val(pred_parsed)
    pred_parsed = rebuild_sql_col(pred_valid_cols, pred_parsed, key_map)

    exact_match = evaluator.eval_exact_match(pred_parsed, gold_parsed)
    partial_scores = evaluator.partial_scores or {}

    main_components = ["select", "where", "group", "order"]
    f1_values = [partial_scores[component]["f1"] for component in main_components if component in partial_scores]
    average_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return {
        "exact_match": exact_match,
        "partial_scores": partial_scores,
        "average_component_f1": average_f1,
    }


def evaluate_batch(
    predictions: List[Dict[str, Any]],
    tables_path: str,
    tables_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Đánh giá hàng loạt từ danh sách prediction.

    Mỗi phần tử predictions cần có: question, gold_sql, predict_sql, db_id
    Tùy chọn: gold_query_toks, predict_query_toks
    """
    if tables_data is None:
        with open(tables_path, "r", encoding="utf-8") as file:
            tables_data = json.load(file)

    tables_by_id = {entry["db_id"]: entry for entry in tables_data}
    foreign_key_maps = build_foreign_key_map_from_json(tables_path)
    evaluator = Evaluator()

    exact_total = 0
    component_f1_sums = {"select": 0.0, "where": 0.0, "group": 0.0, "order": 0.0}
    component_counts = {"select": 0, "where": 0, "group": 0, "order": 0}
    total = len(predictions)

    for item in predictions:
        db_id = item["db_id"]
        table_entry = tables_by_id[db_id]
        schema = Schema(build_schema_from_tables_entry(table_entry))
        key_map = foreign_key_maps.get(db_id, {})

        gold_query = item.get("gold_sql") or item.get("query", "")
        predict_query = item.get("predict_sql") or item.get("predicted_sql", "")

        result = evaluate_em_f1(
            predict_sql=predict_query,
            gold_sql=gold_query,
            db_schema=schema,
            gold_tokens=item.get("gold_query_toks"),
            predict_tokens=item.get("predict_query_toks"),
            foreign_key_map=key_map,
        )

        exact_total += result["exact_match"]
        for component in component_f1_sums:
            if component in result["partial_scores"]:
                component_f1_sums[component] += result["partial_scores"][component]["f1"]
                component_counts[component] += 1

    metrics = {
        "exact_match": exact_total / total if total else 0.0,
        "select_f1": component_f1_sums["select"] / total if total else 0.0,
        "where_f1": component_f1_sums["where"] / total if total else 0.0,
        "group_f1": component_f1_sums["group"] / total if total else 0.0,
        "order_f1": component_f1_sums["order"] / total if total else 0.0,
    }
    f1_list = [metrics["select_f1"], metrics["where_f1"], metrics["group_f1"], metrics["order_f1"]]
    metrics["average_component_f1"] = sum(f1_list) / len(f1_list)
    metrics["total_samples"] = total

    return metrics
