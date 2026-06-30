"""
Bộ parser SQL kế thừa từ Spider Benchmark, được mở rộng cho ViText2SQL.

ViText2SQL sử dụng tên bảng/cột tiếng Việt có thể chứa khoảng trắng (ví dụ: "id tài_sản",
"bộ_phận của tài_sản"). Parser gốc của Spider dùng NLTK word_tokenize không phù hợp;
module này dùng tokenization theo khoảng trắng và hỗ trợ tên cột nhiều token.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

CLAUSE_KEYWORDS = (
    "select", "from", "where", "group", "order", "limit",
    "intersect", "union", "except",
)
JOIN_KEYWORDS = ("join", "on", "as")

WHERE_OPS = (
    "not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {"sql": "sql", "table_unit": "table_unit"}

COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")

# Tập token dừng khi ghép tên cột/bảng nhiều từ
_STOP_TOKENS = set(CLAUSE_KEYWORDS) | set(JOIN_KEYWORDS) | set(WHERE_OPS) | set(
    UNIT_OPS
) | set(AGG_OPS) | set(COND_OPS) | set(SQL_OPS) | set(ORDER_OPS) | {
    ",", "(", ")", ";", "distinct", "by", "having", "limit", "as", "on", "join",
}


class Schema:
    """Ánh xạ tên bảng và cột sang định danh duy nhất (theo Spider)."""

    def __init__(self, schema: Dict[str, List[str]]):
        self._schema = schema
        self._idMap = self._map(schema)

    @property
    def schema(self) -> Dict[str, List[str]]:
        return self._schema

    @property
    def idMap(self) -> Dict[str, str]:
        return self._idMap

    def _map(self, schema: Dict[str, List[str]]) -> Dict[str, str]:
        id_map: Dict[str, str] = {"*": "__all__"}
        for table_name, columns in schema.items():
            normalized_table = normalize_identifier(table_name)
            for column_name in columns:
                key = f"{table_name.lower()}.{column_name.lower()}"
                normalized_key = f"{normalized_table}.{normalize_identifier(column_name)}"
                id_map[key] = f"__{key}__"
                if normalized_key not in id_map:
                    id_map[normalized_key] = f"__{key}__"
            if normalized_table not in id_map:
                id_map[normalized_table] = f"__{table_name.lower()}__"
            id_map[table_name.lower()] = f"__{table_name.lower()}__"
        return id_map


def get_schema_from_sqlite(database_path: str) -> Dict[str, List[str]]:
    """Đọc schema từ file SQLite (định dạng Spider gốc)."""
    schema: Dict[str, List[str]] = {}
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(row[0].lower()) for row in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        schema[table] = [str(column[1].lower()) for column in cursor.fetchall()]
    connection.close()
    return schema


def build_schema_from_tables_entry(entry: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Xây dựng schema từ một mục trong tables.json của ViText2SQL.

    Sử dụng table_names và column_names (tên tiếng Việt) thay vì đọc SQLite.
    """
    schema: Dict[str, List[str]] = {}
    table_names = [name.lower() for name in entry["table_names"]]
    for table_index, table_name in enumerate(table_names):
        columns: List[str] = []
        for column_index, column_name in entry["column_names"]:
            if column_index == -1:
                continue
            if column_index == table_index:
                columns.append(str(column_name).lower())
        schema[table_name] = columns
    return schema


def build_schema_map_from_tables_json(tables_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Tạo dictionary db_id -> schema từ file tables.json."""
    with open(tables_path, "r", encoding="utf-8") as file:
        tables_data = json.load(file)
    return {
        entry["db_id"]: build_schema_from_tables_entry(entry) for entry in tables_data
    }


def normalize_sql_string(sql: str) -> str:
    """
    Chuẩn hóa chuỗi SQL trước khi tokenize.

    - Chuyển về chữ thường
    - Thay nháy đơn bằng nháy kép
    - Thêm khoảng trắng quanh toán tử và dấu ngoặc (theo format ViText2SQL)
    """
    text = str(sql).strip().lower()
    text = text.replace("'", '"')
    text = re.sub(r"([,()=<>!])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_identifier(name: str) -> str:
    """Chuẩn hóa tên bảng/cột để khớp giữa dấu gạch dưới, khoảng trắng và quote."""
    if not isinstance(name, str):
        return ""
    text = name.strip().lower()
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        text = text[1:-1].strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_vitext2sql(string: Union[str, List[str]]) -> List[str]:
    """
    Tokenize câu SQL theo định dạng ViText2SQL.

    Nếu đầu vào đã là danh sách token (query_toks), trả về trực tiếp sau chuẩn hóa.
    """
    if isinstance(string, list):
        return [token.lower() if token not in ('"', "'") else token for token in string]

    text = normalize_sql_string(string)
    token_pattern = r'"[^"]*"(?:\.[^\s,()=<>!]+)*|!=|>=|<=|[=<>(),]|[^\s,()=<>!]+'
    tokens = re.findall(token_pattern, text)
    return [token.lower() for token in tokens if token.strip()]


def _find_table_name(
    tokens: List[str], start_index: int, schema: Dict[str, List[str]]
) -> Tuple[str, int]:
    """Tìm tên bảng dài nhất hợp lệ trong schema, bắt đầu tại start_index."""
    normalized_schema = {
        normalize_identifier(table_name): table_name for table_name in schema
    }
    best_match: Optional[str] = None
    best_end = start_index
    end_index = start_index

    while end_index < len(tokens):
        token = tokens[end_index]
        if token in CLAUSE_KEYWORDS or token in JOIN_KEYWORDS or token in (")", ";", ","):
            break
        candidate = normalize_identifier(" ".join(tokens[start_index : end_index + 1]))
        if candidate in normalized_schema:
            best_match = normalized_schema[candidate]
            best_end = end_index + 1
        end_index += 1

    if best_match is not None:
        return best_match, best_end

    table_parts: List[str] = [tokens[start_index]]
    end_index = start_index + 1
    while (
        end_index < len(tokens)
        and tokens[end_index] != "as"
        and tokens[end_index] not in JOIN_KEYWORDS
        and tokens[end_index] not in CLAUSE_KEYWORDS
        and tokens[end_index] not in (")", ";", ",")
    ):
        table_parts.append(tokens[end_index])
        end_index += 1
    return normalize_identifier(" ".join(table_parts)), end_index


def get_tables_with_alias(
    schema: Dict[str, List[str]], tokens: List[str]
) -> Dict[str, str]:
    """Khởi tạo bản đồ tên bảng chuẩn hóa -> tên bảng gốc."""
    return {
        normalize_identifier(table_name): table_name
        for table_name in schema
    }


def _is_stop_token(token: str) -> bool:
    return token in _STOP_TOKENS


def _resolve_column_key(
    alias_or_table: str,
    column_parts: List[str],
    tables_with_alias: Dict[str, str],
    schema: Schema,
) -> str:
    """Ghép alias/bảng với tên cột (có thể nhiều token) thành khóa schema."""
    table_key = normalize_identifier(alias_or_table)
    table = tables_with_alias.get(table_key, alias_or_table)
    column = normalize_identifier(" ".join(column_parts))
    key = f"{table.lower()}.{column}"
    if key in schema.idMap:
        return schema.idMap[key]
    normalized_key = f"{normalize_identifier(table)}.{column}"
    if normalized_key in schema.idMap:
        return schema.idMap[normalized_key]
    # Thử khớp mờ: cột có thể thiếu prefix bảng
    for candidate_key in schema.idMap:
        if candidate_key.endswith(f".{column}"):
            return schema.idMap[candidate_key]
    raise ValueError(f"Không tìm thấy cột trong schema: {key}")


def parse_col(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, str]:
    """Phân tích một cột; hỗ trợ tên cột tiếng Việt nhiều từ."""
    token = tokens[start_index]
    if token == "*":
        return start_index + 1, schema.idMap[token]

    if "." in token:
        alias, first_col_part = token.split(".", 1)
        column_parts = [first_col_part] if first_col_part else []
        index = start_index + 1
        while index < len(tokens) and not _is_stop_token(tokens[index]):
            column_parts.append(tokens[index])
            index += 1
        col_id = _resolve_column_key(alias, column_parts, tables_with_alias, schema)
        return index, col_id

    if default_tables:
        for alias in default_tables:
            table = tables_with_alias.get(normalize_identifier(alias), alias)
            column_parts = [token]
            index = start_index + 1
            while index < len(tokens) and not _is_stop_token(tokens[index]):
                column_parts.append(tokens[index])
                index += 1
            column_name = " ".join(column_parts)
            normalized_column = normalize_identifier(column_name)
            for column in schema.schema.get(table, []):
                if normalize_identifier(column) == normalized_column:
                    key = f"{table}.{column}"
                    return index, schema.idMap[key]

    raise ValueError(f"Lỗi parse cột: {token}")


def parse_col_unit(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[int, str, bool]]:
    """Phân tích col_unit: (agg_id, col_id, isDistinct)."""
    index = start_index
    length = len(tokens)
    is_block = False
    is_distinct = False

    if tokens[index] == "(":
        is_block = True
        index += 1

    if tokens[index] in AGG_OPS:
        agg_id = AGG_OPS.index(tokens[index])
        index += 1
        if index < length and tokens[index] == "(":
            index += 1
        if index < length and tokens[index] == "distinct":
            index += 1
            is_distinct = True
        index, col_id = parse_col(tokens, index, tables_with_alias, schema, default_tables)
        if index < length and tokens[index] == ")":
            index += 1
        return index, (agg_id, col_id, is_distinct)

    if tokens[index] == "distinct":
        index += 1
        is_distinct = True
    agg_id = AGG_OPS.index("none")
    index, col_id = parse_col(tokens, index, tables_with_alias, schema, default_tables)

    if is_block and index < length and tokens[index] == ")":
        index += 1

    return index, (agg_id, col_id, is_distinct)


def parse_val_unit(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[int, Any, Any]]:
    index = start_index
    length = len(tokens)
    is_block = tokens[index] == "("
    if is_block:
        index += 1

    index, col_unit1 = parse_col_unit(
        tokens, index, tables_with_alias, schema, default_tables
    )
    unit_op = UNIT_OPS.index("none")
    col_unit2 = None

    if index < length and tokens[index] in UNIT_OPS:
        unit_op = UNIT_OPS.index(tokens[index])
        index += 1
        index, col_unit2 = parse_col_unit(
            tokens, index, tables_with_alias, schema, default_tables
        )

    if is_block and index < length and tokens[index] == ")":
        index += 1

    return index, (unit_op, col_unit1, col_unit2)


def parse_table_unit(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
) -> Tuple[int, str, str]:
    """Phân tích tên bảng (có thể nhiều token trước alias)."""
    index = start_index
    table_name, index = _find_table_name(tokens, index, schema.schema)
    if table_name not in schema.idMap:
        raise ValueError(f"Không tìm thấy bảng trong schema: {table_name}")

    normalized_table = normalize_identifier(table_name)
    if normalized_table not in tables_with_alias:
        tables_with_alias[normalized_table] = table_name

    alias: Optional[str] = None
    if index < len(tokens) and tokens[index] == "as":
        if index + 1 < len(tokens):
            alias = tokens[index + 1]
        index += 2
    elif (
        index < len(tokens)
        and tokens[index] not in CLAUSE_KEYWORDS
        and tokens[index] not in JOIN_KEYWORDS
        and tokens[index] not in (")", ";", ",")
    ):
        alias = tokens[index]
        index += 1

    if alias:
        tables_with_alias[normalize_identifier(alias)] = table_name

    return index, schema.idMap[table_name], table_name


def parse_value(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Any]:
    index = start_index
    length = len(tokens)
    is_block = tokens[index] == "("
    if is_block:
        index += 1

    if tokens[index] == "select":
        index, value = parse_sql(tokens, index, tables_with_alias, schema)
    elif '"' in tokens[index]:
        value = tokens[index]
        index += 1
    elif tokens[index] == "null":
        value = "null"
        index += 1
    elif tokens[index] == "not" and index + 1 < length and tokens[index + 1] == "null":
        value = "not null"
        index += 2
    else:
        try:
            value = float(tokens[index])
            index += 1
        except ValueError:
            end_index = index
            while (
                end_index < length
                and tokens[end_index] not in {",", ")", "and"}
                and tokens[end_index] not in CLAUSE_KEYWORDS
                and tokens[end_index] not in JOIN_KEYWORDS
            ):
                end_index += 1
            index, value = parse_col_unit(
                tokens[start_index:end_index], 0, tables_with_alias, schema, default_tables
            )
            index = end_index

    if is_block and index < length and tokens[index] == ")":
        index += 1

    return index, value


def parse_condition(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, List[Any]]:
    index = start_index
    length = len(tokens)
    conditions: List[Any] = []

    while index < length:
        index, val_unit = parse_val_unit(
            tokens, index, tables_with_alias, schema, default_tables
        )
        not_op = False
        if index < length and tokens[index] == "not":
            not_op = True
            index += 1

        if index >= length or tokens[index] not in WHERE_OPS:
            break

        op_id = WHERE_OPS.index(tokens[index])
        index += 1
        val1 = val2 = None

        if op_id == WHERE_OPS.index("between"):
            index, val1 = parse_value(tokens, index, tables_with_alias, schema, default_tables)
            if index < length and tokens[index] == "and":
                index += 1
            index, val2 = parse_value(tokens, index, tables_with_alias, schema, default_tables)
        else:
            index, val1 = parse_value(tokens, index, tables_with_alias, schema, default_tables)

        conditions.append((not_op, op_id, val_unit, val1, val2))

        if index < length and (
            tokens[index] in CLAUSE_KEYWORDS
            or tokens[index] in (")", ";")
            or tokens[index] in JOIN_KEYWORDS
        ):
            break

        if index < length and tokens[index] in COND_OPS:
            conditions.append(tokens[index])
            index += 1
        else:
            break

    return index, conditions


def parse_select(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[bool, List[Any]]]:
    index = start_index
    assert tokens[index] == "select"
    index += 1
    is_distinct = False
    if index < len(tokens) and tokens[index] == "distinct":
        index += 1
        is_distinct = True

    val_units: List[Any] = []
    while index < len(tokens) and tokens[index] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if tokens[index] in AGG_OPS:
            agg_id = AGG_OPS.index(tokens[index])
            index += 1
        index, val_unit = parse_val_unit(
            tokens, index, tables_with_alias, schema, default_tables
        )
        val_units.append((agg_id, val_unit))
        if index < len(tokens) and tokens[index] == "as":
            index += 1
            if index < len(tokens) and tokens[index] not in CLAUSE_KEYWORDS + (",",):
                index += 1
        if index < len(tokens) and tokens[index] == ",":
            index += 1

    return index, (is_distinct, val_units)


def parse_from(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
) -> Tuple[int, List[Any], List[Any], List[str]]:
    assert "from" in tokens[start_index:]
    length = len(tokens)
    index = tokens.index("from", start_index) + 1
    default_tables: List[str] = []
    table_units: List[Any] = []
    conditions: List[Any] = []

    while index < length:
        is_block = tokens[index] == "("
        if is_block:
            index += 1

        if tokens[index] == "select":
            index, sub_sql = parse_sql(tokens, index, tables_with_alias, schema)
            table_units.append((TABLE_TYPE["sql"], sub_sql))
        else:
            if index < length and tokens[index] == "join":
                index += 1
            index, table_unit, table_name = parse_table_unit(
                tokens, index, tables_with_alias, schema
            )
            table_units.append((TABLE_TYPE["table_unit"], table_unit))
            default_tables.append(table_name)

            if index < length and tokens[index] == "on":
                index += 1
                index, join_conds = parse_condition(
                    tokens, index, tables_with_alias, schema, default_tables
                )
                if conditions:
                    conditions.append("and")
                conditions.extend(join_conds)

        if is_block and index < length and tokens[index] == ")":
            index += 1

        if index < length and (
            tokens[index] in CLAUSE_KEYWORDS or tokens[index] in (")", ";")
        ):
            break

    return index, table_units, conditions, default_tables


def parse_where(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, List[Any]]:
    index = start_index
    if index >= len(tokens) or tokens[index] != "where":
        return index, []
    index += 1
    index, conditions = parse_condition(
        tokens, index, tables_with_alias, schema, default_tables
    )
    return index, conditions


def parse_group_by(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, List[Any]]:
    index = start_index
    col_units: List[Any] = []
    if index >= len(tokens) or tokens[index] != "group":
        return index, col_units
    index += 2  # bỏ 'group by'
    while index < len(tokens) and tokens[index] not in CLAUSE_KEYWORDS + (")", ";"):
        index, col_unit = parse_col_unit(
            tokens, index, tables_with_alias, schema, default_tables
        )
        col_units.append(col_unit)
        if index < len(tokens) and tokens[index] == ",":
            index += 1
        else:
            break
    return index, col_units


def parse_order_by(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[str, List[Any]]]:
    index = start_index
    val_units: List[Any] = []
    order_type = "asc"
    if index >= len(tokens) or tokens[index] != "order":
        return index, (order_type, val_units)
    index += 2  # bỏ 'order by'
    while index < len(tokens) and tokens[index] not in CLAUSE_KEYWORDS + (")", ";"):
        index, val_unit = parse_val_unit(
            tokens, index, tables_with_alias, schema, default_tables
        )
        val_units.append(val_unit)
        if index < len(tokens) and tokens[index] in ORDER_OPS:
            order_type = tokens[index]
            index += 1
        if index < len(tokens) and tokens[index] == ",":
            index += 1
        else:
            break
    return index, (order_type, val_units)


def parse_having(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, List[Any]]:
    index = start_index
    if index >= len(tokens) or tokens[index] != "having":
        return index, []
    index += 1
    index, conditions = parse_condition(
        tokens, index, tables_with_alias, schema, default_tables
    )
    return index, conditions


def parse_limit(tokens: List[str], start_index: int) -> Tuple[int, Optional[int]]:
    index = start_index
    if index < len(tokens) and tokens[index] == "limit":
        index += 1
        limit_value = int(tokens[index])
        index += 1
        return index, limit_value
    return index, None


def skip_semicolon(tokens: List[str], start_index: int) -> int:
    index = start_index
    while index < len(tokens) and tokens[index] == ";":
        index += 1
    return index


def parse_sql(
    tokens: List[str],
    start_index: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
) -> Tuple[int, Dict[str, Any]]:
    is_block = tokens[start_index] == "("
    index = start_index
    if is_block:
        index += 1

    from_end, table_units, from_conds, default_tables = parse_from(
        tokens, start_index, tables_with_alias, schema
    )
    sql: Dict[str, Any] = {
        "from": {"table_units": table_units, "conds": from_conds},
    }

    _, select_col_units = parse_select(
        tokens, index, tables_with_alias, schema, default_tables
    )
    sql["select"] = select_col_units
    index = from_end

    index, where_conds = parse_where(
        tokens, index, tables_with_alias, schema, default_tables
    )
    sql["where"] = where_conds

    index, group_col_units = parse_group_by(
        tokens, index, tables_with_alias, schema, default_tables
    )
    sql["groupBy"] = group_col_units

    index, having_conds = parse_having(
        tokens, index, tables_with_alias, schema, default_tables
    )
    sql["having"] = having_conds

    index, order_col_units = parse_order_by(
        tokens, index, tables_with_alias, schema, default_tables
    )
    sql["orderBy"] = order_col_units

    index, limit_val = parse_limit(tokens, index)
    sql["limit"] = limit_val

    index = skip_semicolon(tokens, index)
    if is_block and index < len(tokens) and tokens[index] == ")":
        index += 1
        index = skip_semicolon(tokens, index)

    for op in SQL_OPS:
        sql[op] = None
    if index < len(tokens) and tokens[index] in SQL_OPS:
        sql_op = tokens[index]
        index += 1
        index, nested_sql = parse_sql(tokens, index, tables_with_alias, schema)
        sql[sql_op] = nested_sql

    return index, sql


def get_sql(
    schema: Schema,
    query: Union[str, List[str]],
) -> Dict[str, Any]:
    """
    Chuyển câu SQL (chuỗi hoặc query_toks) thành cấu trúc AST theo Spider.

    Args:
        schema: Đối tượng Schema đã khởi tạo
        query: Câu SQL hoặc danh sách token query_toks
    """
    tokens = tokenize_vitext2sql(query)
    tables_with_alias = get_tables_with_alias(schema.schema, tokens)
    _, parsed = parse_sql(tokens, 0, tables_with_alias, schema)
    return parsed


def empty_sql() -> Dict[str, Any]:
    """Trả về AST SQL rỗng khi parse thất bại."""
    return {
        "except": None,
        "from": {"conds": [], "table_units": []},
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [False, []],
        "union": None,
        "where": [],
    }
