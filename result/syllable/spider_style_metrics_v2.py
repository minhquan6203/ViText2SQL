import json
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# 1. Load schema
# ---------------------------------------------------------------------------

def load_schema(tables_path):
    tables = json.load(open(tables_path, encoding="utf-8"))
    db_map = {}
    for t in tables:
        db_id = t["db_id"]
        table_names = t["table_names_original"]
        col_entries = t["column_names_original"]  # [(table_idx, col_name), ...], idx0 = (-1,'*')

        # per-table column name list (word-tuples), sorted longest-first for greedy match
        table_columns = defaultdict(list)  # table_idx -> list of (word_tuple, col_name_lower)
        all_columns = []  # list of (word_tuple, col_name_lower) across all tables
        for tbl_idx, col_name in col_entries:
            if tbl_idx == -1:
                continue
            words = tuple(col_name.lower().split())
            table_columns[tbl_idx].append((words, col_name.lower()))
            all_columns.append((words, col_name.lower()))

        for tbl_idx in table_columns:
            table_columns[tbl_idx].sort(key=lambda x: -len(x[0]))
        all_columns.sort(key=lambda x: -len(x[0]))

        table_word_tuples = [(tuple(tn.lower().split()), i, tn.lower())
                              for i, tn in enumerate(table_names)]
        table_word_tuples.sort(key=lambda x: -len(x[0]))

        db_map[db_id] = {
            "table_names": [tn.lower() for tn in table_names],
            "table_word_tuples": table_word_tuples,
            "table_columns": table_columns,   # table_idx -> sorted list
            "all_columns": all_columns,       # sorted list across db
        }
    return db_map


# ---------------------------------------------------------------------------
# 2. Tokenize raw SQL string (Khử nhiễu dấu ngoặc kép \")
# ---------------------------------------------------------------------------

def raw_tokenize(sql):
    sql = sql.strip().lower()
    # Loại bỏ dấu ngoặc kép dư thừa do model sinh ra (e.g. \"tình trạng lỗi\" -> tình trạng lỗi)
    sql = sql.replace('"', '').replace('\\', '')
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*([(),])\s*", r" \1 ", sql)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql.split(" ") if sql else []


SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "having", "order", "limit",
    "join", "on", "as", "and", "or", "not", "in", "like", "between", "is",
    "null", "exists", "desc", "asc", "intersect", "union", "except",
    "distinct", "count", "sum", "avg", "max", "min", ",", "(", ")",
    "=", "<", ">", "<=", ">=", "!=", "<>", "*",
}


# ---------------------------------------------------------------------------
# 3. Schema-aware canonicalisation
# ---------------------------------------------------------------------------

def greedy_match(tokens, start, candidates):
    for word_tuple, name in candidates:
        n = len(word_tuple)
        if start + n > len(tokens):
            continue
        if tuple(tokens[start:start + n]) == word_tuple:
            return name, n
    return None, 0


def split_top_level_blocks(tokens):
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


def canonicalize(sql, db_schema):
    tokens = raw_tokenize(sql)
    out_all = []
    for block_tokens, sep in split_top_level_blocks(tokens):
        out_all.extend(_canonicalize_block(block_tokens, db_schema))
        if sep is not None:
            out_all.append(sep)
    return out_all


def _canonicalize_block(tokens, db_schema):
    n = len(tokens)
    out = []
    alias_map = {}  # alias -> table_name

    # ---- pass 1: scan FROM/JOIN to build alias_map -------------------
    j = 0
    while j < n:
        tok = tokens[j]
        if tok in ("from", "join"):
            j += 1
            tbl_name, nwords = greedy_match(tokens, j, db_schema["table_word_tuples"] and
                                             [(wt, tn) for wt, _, tn in db_schema["table_word_tuples"]])
            if tbl_name is not None:
                j += nwords
                alias = tbl_name
                if j < n and tokens[j] == "as":
                    j += 1
                    if j < n:
                        alias = tokens[j]
                        alias_map[alias] = tbl_name
                        j += 1
                elif j < n and tokens[j] not in SQL_KEYWORDS and "." not in tokens[j]:
                    alias = tokens[j]
                    alias_map[alias] = tbl_name
                    j += 1
                alias_map[tbl_name] = tbl_name
                continue
            else:
                j += 1
                continue
        j += 1

    # ---- pass 2: rebuild token stream, resolving columns/tables ------
    i = 0
    while i < n:
        tok = tokens[i]

        if tok in ("from", "join"):
            out.append(tok)
            i += 1
            tbl_name, nwords = greedy_match(
                tokens, i, [(wt, tn) for wt, _, tn in db_schema["table_word_tuples"]])
            if tbl_name is not None:
                out.append(tbl_name.replace(" ", "_"))
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
                    next((idx for idx, nm in enumerate(db_schema["table_names"]) if nm == table_name), -1),
                    [])
            else:
                candidates = db_schema["all_columns"]

            probe = [first_word] + tokens[i + 1:i + 8]
            col_name, nwords = greedy_match(probe, 0, candidates)
            if col_name is not None:
                canon_table = table_name if table_name is not None else "unk"
                out.append(f"{canon_table.replace(' ', '_')}.{col_name.replace(' ', '_')}")
                i += nwords
                continue
            else:
                out.append(tok)
                i += 1
                continue

        if tok not in SQL_KEYWORDS and re.match(r"^[a-zA-ZÀ-ỹ_]", tok):
            probe = [tok] + tokens[i + 1:i + 8]
            col_name, nwords = greedy_match(probe, 0, db_schema["all_columns"])
            if col_name is not None and nwords > 0:
                out.append(f"?.{col_name.replace(' ', '_')}")
                i += nwords
                continue

        out.append(tok)
        i += 1

    return out


# ---------------------------------------------------------------------------
# 4. Order-agnostic Logic (Xử lý Reorder nâng cao)
# ---------------------------------------------------------------------------

def split_clauses(canon_tokens):
    depth = 0
    current = None
    clauses = defaultdict(list)
    i, n = 0, len(canon_tokens)
    while i < n:
        tok = canon_tokens[i]
        if tok == "(":
            depth += 1
            if current:
                clauses[current].append(tok)
            i += 1
            continue
        if tok == ")":
            depth -= 1
            if current:
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
                current = {"select": "SELECT", "where": "WHERE", "having": "WHERE",
                           "from": "FROM", "limit": "LIMIT"}[tok]
                i += 1
                continue
            if tok in ("intersect", "union", "except"):
                clauses["KEYWORDS"].append(tok)
                i += 1
                continue
        if current is not None:
            clauses[current].append(tok)
        OTHER_KEYWORDS = {"join", "distinct", "and", "or", "not", "in", "like",
                           "between", "is", "null", "exists", "as", "count",
                           "sum", "avg", "max", "min", "desc", "asc"}
        if tok in OTHER_KEYWORDS:
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


COMPONENTS = ["SELECT", "WHERE", "GROUP BY", "ORDER BY", "KEYWORDS"]


def component_items(clauses):
    result = {}
    result["SELECT"] = set(split_items(clauses.get("SELECT", []), {","}))
    result["WHERE"] = set(split_items(clauses.get("WHERE", []), {"and", "or"}))
    result["GROUP BY"] = set(split_items(clauses.get("GROUP BY", []), {","}))
    result["ORDER BY"] = set(split_items(clauses.get("ORDER BY", []), {","}))
    result["KEYWORDS"] = set(clauses.get("KEYWORDS", []))
    
    # Đóng gói cả thông tin LIMIT vào KEYWORDS/SELECT dưới dạng tập hợp để tăng độ chuẩn xác khi EM
    if "LIMIT" in clauses:
        result["KEYWORDS"].add("limit")
        result["KEYWORDS"].update(clauses["LIMIT"])
        
    return result


def is_structurally_equal(gold_items, pred_items):
    """
    So sánh hai câu SQL dựa trên cấu trúc tập hợp của từng component.
    Nếu toàn bộ các cặp component giống nhau hoàn toàn (không phụ thuộc thứ tự), trả về True.
    """
    for comp in COMPONENTS:
        if gold_items[comp] != pred_items[comp]:
            return False
    return True


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def set_f1(gold_set, pred_set):
    if not gold_set and not pred_set:
        return None
    overlap = len(gold_set & pred_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(examples, db_map):
    n = len(examples)
    em_correct = 0
    component_scores = {c: [] for c in COMPONENTS}
    unknown_db = 0

    for ex in examples:
        db_id = ex.get("db_id")
        gold_sql = ex["gold_sql"]
        pred_sql = ex.get("predict_sql", "")
        db_schema = db_map.get(db_id)

        if db_schema is None:
            unknown_db += 1
            gold_canon = raw_tokenize(gold_sql)
            pred_canon = raw_tokenize(pred_sql)
        else:
            gold_canon = canonicalize(gold_sql, db_schema)
            pred_canon = canonicalize(pred_sql, db_schema)

        gold_items = component_items(split_clauses(gold_canon))
        pred_items = component_items(split_clauses(pred_canon))

        # Thay vì so khớp chuỗi thô (dễ lệch vị trí), ta kiểm tra độ tương thích cấu trúc Set
        if is_structurally_equal(gold_items, pred_items):
            em_correct += 1

        for comp in COMPONENTS:
            f1 = set_f1(gold_items[comp], pred_items[comp])
            if f1 is not None:
                component_scores[comp].append(f1)

    result = {
        "n_examples": n,
        "unknown_db": unknown_db,
        "exact_match_accuracy": 100.0 * em_correct / n if n else 0.0,
    }
    for comp in COMPONENTS:
        scores = component_scores[comp]
        result[f"F1_{comp}"] = 100.0 * sum(scores) / len(scores) if scores else float("nan")
    return result


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main(tables_path, pred_paths):
    db_map = load_schema(tables_path)
    rows = []
    for path in pred_paths:
        examples = json.load(open(path, encoding="utf-8"))
        res = evaluate(examples, db_map)
        res["model"] = examples[0].get("model", "?") if examples else "?"
        res["mode"] = examples[0].get("mode", "?") if examples else "?"
        rows.append(res)

    header = ["model", "mode", "n_examples", "exact_match_accuracy"] + [f"F1_{c}" for c in COMPONENTS]
    widths = {h: 22 if h == "model" else 14 for h in header}

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    line = " | ".join(h.ljust(widths[h]) for h in header)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(fmt(r[h]).ljust(widths[h]) for h in header))
        if r["unknown_db"]:
            print(f"  (canh bao: {r['unknown_db']} cau khong tim thay db_id trong tables.json)")

    return rows


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cach dung: python3 spider_style_metrics_v2.py tables.json pred1.txt pred2.txt ...")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2:])