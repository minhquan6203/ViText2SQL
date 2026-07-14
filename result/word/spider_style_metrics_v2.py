"""
Tinh metric giong paper Nguyen et al. 2020 - "A Pilot Study of Text-to-SQL
Semantic Parsing for Vietnamese" (EMNLP Findings 2020), co su dung schema
(tables.json) de resolve alias -> ten bang/cot that, giup so khop chinh xac
hon ban khong co schema.

2 metric (theo Yu et al. 2018 - Spider):
  1. Exact Matching Accuracy (EM)
  2. Component Matching F1 cho SELECT / WHERE / GROUP BY / ORDER BY / KEYWORDS

Cach dung:
    python3 spider_style_metrics_v2.py tables.json pred1.txt pred2.txt ...

GIOI HAN con lai (khong the khac phuc neu khong co file database that / khong
viet lai toan bo process_sql.py cua Spider):
  - Khong tinh duoc Execution Accuracy (can file .sqlite that de chay query).
  - Component matching o day dua tren tach "item" theo dau phay / AND-OR va so
    sanh tap hop (set) cac item da duoc chuan hoa alias->ten that, thay vi
    parse thanh AST day du nhu process_sql.py goc (vi du: chua xu ly rieng
    aggregation + DISTINCT thanh tuple co cau truc, chua tach nested subquery
    thanh cay de so sanh de quy). Day la xap xi manh, khong phai ban chinh xac
    100% cua eval script goc, nhung da loai bo duoc nguon nhieu lon nhat la
    alias khac nhau (t1 vs t) va tu nhieu-tu tieng Viet bi tach sai.
"""

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
# 2. Tokenize raw SQL string (whitespace-based, keep punctuation as separate
#    tokens -- data is already formatted with spaces around ( ) , = < > etc.)
# ---------------------------------------------------------------------------

def raw_tokenize(sql):
    sql = sql.strip().lower()
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
# 3. Schema-aware canonicalisation: replace "alias.word word word" or bare
#    "word word word" column references with canonical "table.column" names,
#    and multi-word table names in FROM/JOIN with their canonical form. This
#    removes alias-naming noise (t1 vs t) before we compare gold vs predict.
# ---------------------------------------------------------------------------

def greedy_match(tokens, start, candidates):
    """
    candidates: list of (word_tuple, name) sorted longest-first.
    Try to match the longest word_tuple starting at tokens[start:].
    Return (matched_name, n_words) or (None, 0).
    """
    for word_tuple, name in candidates:
        n = len(word_tuple)
        if start + n > len(tokens):
            continue
        if tuple(tokens[start:start + n]) == word_tuple:
            return name, n
    return None, 0


def split_top_level_blocks(tokens):
    """
    Split tokens at top-level (paren depth 0) occurrences of intersect/union/except.
    Each block gets its own alias scope (aliases like t1/t2 are commonly reused
    independently in each block of a set-operator query). Returns list of
    (block_tokens, following_separator_or_None).
    """
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
    """
    Returns list of canonical tokens:
      - qualified column refs -> "table_name.column_name" (single token, spaces->'_')
      - table refs in FROM/JOIN -> table_name (single token, spaces->'_'), alias tokens dropped
      - everything else kept as-is (lowercased)
    Query is split at top-level INTERSECT/UNION/EXCEPT into independent blocks,
    each with its own alias scope, before resolving.
    """
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

    i = 0
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
                alias = tbl_name  # default: alias == table name if none given
                if j < n and tokens[j] == "as":
                    j += 1
                    if j < n:
                        alias = tokens[j]
                        alias_map[alias] = tbl_name
                        j += 1
                elif j < n and tokens[j] not in SQL_KEYWORDS and "." not in tokens[j]:
                    # implicit alias, e.g. "from tài sản t"
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
                        i += 1  # skip alias token entirely (already canonical via table name)
                elif i < n and tokens[i] not in SQL_KEYWORDS and "." not in tokens[i]:
                    i += 1  # skip implicit alias token
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

            probe = [first_word] + tokens[i + 1:i + 8]  # look ahead up to 7 more words
            col_name, nwords = greedy_match(probe, 0, candidates)
            if col_name is not None:
                canon_table = table_name if table_name is not None else "unk"
                out.append(f"{canon_table.replace(' ', '_')}.{col_name.replace(' ', '_')}")
                i += nwords  # first_word + (nwords-1) following tokens
                continue
            else:
                out.append(tok)
                i += 1
                continue

        # bare (unqualified) column reference, e.g. after GROUP BY / ORDER BY / SELECT
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
# 4. Clause splitting (top-level, paren-depth aware) on canonical tokens
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
    """Split a token list into items at top-level (depth 0) occurrences of seps."""
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
    """Convert raw clause token lists into a dict of component -> set(items)."""
    result = {}
    result["SELECT"] = set(split_items(clauses.get("SELECT", []), {","}))
    result["WHERE"] = set(split_items(clauses.get("WHERE", []), {"and", "or"}))
    result["GROUP BY"] = set(split_items(clauses.get("GROUP BY", []), {","}))
    result["ORDER BY"] = set(split_items(clauses.get("ORDER BY", []), {","}))
    result["KEYWORDS"] = set(clauses.get("KEYWORDS", []))
    return result


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

        if " ".join(gold_canon) == " ".join(pred_canon):
            em_correct += 1

        gold_items = component_items(split_clauses(gold_canon))
        pred_items = component_items(split_clauses(pred_canon))

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
