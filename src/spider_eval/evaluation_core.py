import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

COMPONENTS = ["SELECT", "WHERE", "GROUP BY", "ORDER BY", "KEYWORDS"]
SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "having", "order", "limit",
    "join", "on", "as", "and", "or", "not", "in", "like", "between", "is",
    "null", "exists", "desc", "asc", "intersect", "union", "except",
    "distinct", "count", "sum", "avg", "max", "min", ",", "(", ")",
    "=", "<", ">", "<=", ">=", "!=", "<>", "*",
}


def load_schema(tables_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load schema map keyed by db_id."""
    with open(tables_path, "r", encoding="utf-8") as fh:
        tables = json.load(fh)

    db_map: Dict[str, Dict[str, Any]] = {}
    for table_entry in tables:
        db_id = table_entry["db_id"]
        table_names = table_entry.get("table_names_original", table_entry.get("table_names", []))
        col_entries = table_entry.get("column_names_original", table_entry.get("column_names", []))

        table_columns: Dict[int, List[tuple]] = defaultdict(list)
        all_columns: List[tuple] = []
        for tbl_idx, col_name in col_entries:
            if tbl_idx == -1:
                continue
            words = tuple(str(col_name).lower().split())
            table_columns[tbl_idx].append((words, str(col_name).lower()))
            all_columns.append((words, str(col_name).lower()))

        for tbl_idx in table_columns:
            table_columns[tbl_idx].sort(key=lambda x: -len(x[0]))
        all_columns.sort(key=lambda x: -len(x[0]))

        table_word_tuples = [
            (tuple(str(name).lower().split()), i, str(name).lower())
            for i, name in enumerate(table_names)
        ]
        table_word_tuples.sort(key=lambda x: -len(x[0]))

        db_map[db_id] = {
            "table_names": [str(name).lower() for name in table_names],
            "table_word_tuples": table_word_tuples,
            "table_columns": table_columns,
            "all_columns": all_columns,
        }
    return db_map


def raw_tokenize(sql: str) -> List[str]:
    if sql is None:
        return []
    sql = str(sql).strip().lower()
    sql = sql.replace('"', "").replace("\\", "")
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*([(),])\s*", r" \1 ", sql)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql.split(" ") if sql else []


def greedy_match(tokens: List[str], start: int, candidates):
    for word_tuple, name in candidates:
        n = len(word_tuple)
        if start + n > len(tokens):
            continue
        if tuple(tokens[start : start + n]) == word_tuple:
            return name, n
    return None, 0


def split_top_level_blocks(tokens: List[str]):
    cur = []
    depth = 0
    result = []
    for tok in tokens:
        if tok == "(":
            depth += 1
            cur.append(tok)
            continue
        if tok == ")":
            depth -= 1
            cur.append(tok)
            continue
        if depth == 0 and tok in ("intersect", "union", "except"):
            result.append((cur, tok))
            cur = []
            continue
        cur.append(tok)
    result.append((cur, None))
    return result


def canonicalize(sql: str, db_schema: Dict[str, Any]) -> List[str]:
    tokens = raw_tokenize(sql)
    out_all: List[str] = []
    for block_tokens, sep in split_top_level_blocks(tokens):
        out_all.extend(_canonicalize_block(block_tokens, db_schema))
        if sep is not None:
            out_all.append(sep)
    return out_all


def _canonicalize_block(tokens: List[str], db_schema: Dict[str, Any]) -> List[str]:
    n = len(tokens)
    out: List[str] = []
    alias_map: Dict[str, str] = {}

    j = 0
    while j < n:
        tok = tokens[j]
        if tok in ("from", "join"):
            j += 1
            table_name, nwords = greedy_match(
                tokens,
                j,
                [(wt, tn) for wt, _, tn in db_schema["table_word_tuples"]],
            )
            if table_name is not None:
                j += nwords
                alias = table_name
                if j < n and tokens[j] == "as":
                    j += 1
                    if j < n:
                        alias = tokens[j]
                        alias_map[alias] = table_name
                        j += 1
                elif j < n and tokens[j] not in SQL_KEYWORDS and "." not in tokens[j]:
                    alias = tokens[j]
                    alias_map[alias] = table_name
                    j += 1
                alias_map[table_name] = table_name
                continue
            j += 1
            continue
        j += 1

    i = 0
    while i < n:
        tok = tokens[i]

        if tok in ("from", "join"):
            out.append(tok)
            i += 1
            table_name, nwords = greedy_match(
                tokens,
                i,
                [(wt, tn) for wt, _, tn in db_schema["table_word_tuples"]],
            )
            if table_name is not None:
                out.append(table_name.replace(" ", "_"))
                i += nwords
                if i < n and tokens[i] == "as":
                    i += 1
                    if i < n:
                        i += 1
                elif i < n and tokens[i] not in SQL_KEYWORDS and "." not in tokens[i]:
                    i += 1
            continue

        if "." in tok and not re.match(r"^\d+\.\d+$", tok):
            prefix, first_word = tok.split(".", 1)
            table_name = alias_map.get(prefix)
            if table_name is not None:
                candidates = db_schema["table_columns"].get(
                    next((idx for idx, name in enumerate(db_schema["table_names"]) if name == table_name), -1),
                    [],
                )
            else:
                candidates = db_schema["all_columns"]

            probe = [first_word] + tokens[i + 1 : i + 8]
            col_name, nwords = greedy_match(probe, 0, candidates)
            if col_name is not None:
                canon_table = table_name if table_name is not None else "unk"
                out.append(f"{canon_table.replace(' ', '_')}.{col_name.replace(' ', '_')}")
                i += nwords
                continue
            out.append(tok)
            i += 1
            continue

        if tok not in SQL_KEYWORDS and re.match(r"^[a-zA-ZÀ-ỹ_]", tok):
            probe = [tok] + tokens[i + 1 : i + 8]
            col_name, nwords = greedy_match(probe, 0, db_schema["all_columns"])
            if col_name is not None and nwords > 0:
                out.append(f"?.{col_name.replace(' ', '_')}")
                i += nwords
                continue

        out.append(tok)
        i += 1

    return out


def split_clauses(canon_tokens: List[str]):
    depth = 0
    current = None
    clauses: Dict[str, List[str]] = defaultdict(list)
    i, n = 0, len(canon_tokens)
    while i < n:
        tok = canon_tokens[i]
        if tok == "(":
            depth += 1
            if current is not None:
                clauses[current].append(tok)
            i += 1
            continue
        if tok == ")":
            depth -= 1
            if current is not None:
                clauses[current].append(tok)
            i += 1
            continue
        if depth == 0:
            if tok == "group" and i + 1 < n and canon_tokens[i + 1] == "by":
                current = "GROUP BY"
                i += 2
                continue
            if tok == "order" and i + 1 < n and canon_tokens[i + 1] == "by":
                current = "ORDER BY"
                i += 2
                continue
            if tok in ("select", "where", "having", "from", "limit"):
                current = {"select": "SELECT", "where": "WHERE", "having": "WHERE", "from": "FROM", "limit": "LIMIT"}[tok]
                i += 1
                continue
            if tok in ("intersect", "union", "except"):
                clauses["KEYWORDS"].append(tok)
                i += 1
                continue
        if current is not None:
            clauses[current].append(tok)
        other_keywords = {"join", "distinct", "and", "or", "not", "in", "like", "between", "is", "null", "exists", "as", "count", "sum", "avg", "max", "min", "desc", "asc"}
        if tok in other_keywords:
            clauses["KEYWORDS"].append(tok)
        i += 1
    return clauses


def split_items(tokens, seps):
    items, cur, depth = [], [], 0
    for tok in tokens:
        if tok == "(":
            depth += 1
            cur.append(tok)
            continue
        if tok == ")":
            depth -= 1
            cur.append(tok)
            continue
        if depth == 0 and tok in seps:
            if cur:
                items.append(" ".join(cur))
            cur = []
            continue
        cur.append(tok)
    if cur:
        items.append(" ".join(cur))
    return items


def component_items(clauses):
    result = {}
    result["SELECT"] = set(split_items(clauses.get("SELECT", []), {","}))
    result["WHERE"] = set(split_items(clauses.get("WHERE", []), {"and", "or"}))
    result["GROUP BY"] = set(split_items(clauses.get("GROUP BY", []), {","}))
    result["ORDER BY"] = set(split_items(clauses.get("ORDER BY", []), {","}))
    result["KEYWORDS"] = set(clauses.get("KEYWORDS", []))
    if "LIMIT" in clauses:
        result["KEYWORDS"].add("limit")
        result["KEYWORDS"].update(clauses["LIMIT"])
    return result


def is_structurally_equal(gold_items, pred_items):
    for comp in COMPONENTS:
        if gold_items[comp] != pred_items[comp]:
            return False
    return True


def set_f1(gold_set, pred_set):
    if not gold_set and not pred_set:
        return None
    overlap = len(gold_set & pred_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_records(records: Iterable[Dict[str, Any]], db_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    n = len(records)
    em_correct = 0
    f1_scores: List[float] = []
    component_scores = {c: [] for c in COMPONENTS}
    unknown_db = 0

    for ex in records:
        db_id = ex.get("db_id")
        gold_sql = ex.get("gold_sql") or ""
        pred_sql = ex.get("predict_sql") or ""
        db_schema = db_map.get(db_id)

        if db_schema is None:
            unknown_db += 1
            gold_canon = raw_tokenize(gold_sql)
            pred_canon = raw_tokenize(pred_sql)
        else:
            gold_canon = canonicalize(gold_sql, db_schema)
            pred_canon = canonicalize(pred_sql, db_schema)

        gold_set = set(gold_canon)
        pred_set = set(pred_canon)

        if gold_canon == pred_canon:
            em_correct += 1

        f1 = set_f1(gold_set, pred_set)
        if f1 is not None:
            f1_scores.append(f1)

        gold_items = component_items(split_clauses(gold_canon))
        pred_items = component_items(split_clauses(pred_canon))
        for comp in COMPONENTS:
            score = set_f1(gold_items[comp], pred_items[comp])
            if score is not None:
                component_scores[comp].append(score)

    result = {
        "total_samples": n,
        "unknown_db": unknown_db,
        "exact_match": 100.0 * em_correct / n if n else 0.0,
        "f1": 100.0 * sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }

    if COMPONENTS:
        result["component_f1"] = {
            comp: 100.0 * sum(component_scores[comp]) / len(component_scores[comp]) if component_scores[comp] else 0.0
            for comp in COMPONENTS
        }

    return result


def evaluate_predictions(predictions_path: str | Path, tables_path: str | Path) -> Dict[str, Any]:
    with open(predictions_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = payload.get("predictions", payload.get("records", []))
    db_map = load_schema(tables_path)
    return evaluate_records(payload, db_map)


__all__ = [
    "COMPONENTS",
    "load_schema",
    "raw_tokenize",
    "canonicalize",
    "evaluate_records",
    "evaluate_predictions",
]
