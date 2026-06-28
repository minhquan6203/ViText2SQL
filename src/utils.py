"""
Tiện ích dùng chung: tải dữ liệu, schema linking, và định dạng prompt Text-to-SQL.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlparse
from rank_bm25 import BM25Okapi

# Đường dẫn mặc định
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "word-level"
DEFAULT_TABLES_PATH = DEFAULT_DATA_DIR / "tables.json"
DEFAULT_TRAIN_PATH = DEFAULT_DATA_DIR / "train.json"
DEFAULT_DEV_PATH = DEFAULT_DATA_DIR / "dev.json"
DEFAULT_DATABASE_DIR = PROJECT_ROOT / "data" / "database"

# Đăng ký mô hình LLM phân khúc ~2B hỗ trợ bởi pipeline
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen2.5-coder-1.5b": {
        "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "max_seq_length": 4096,
        "chat_template": True,
    },
    "llama-3.2-3b": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "max_seq_length": 4096,
        "chat_template": True,
    },
    "gemma-2-2b": {
        "model_name": "google/gemma-2-2b-it",
        "max_seq_length": 4096,
        "chat_template": True,
    },
}


def load_json(path: str | Path) -> Any:
    """Đọc file JSON với encoding UTF-8."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Ghi dữ liệu ra file JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def load_vitext2sql_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Tải bộ dữ liệu ViText2SQL: train.json, dev.json, tables.json.

    Returns:
        (train_data, dev_data, tables_data)
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.json"
    dev_path = data_dir / "dev.json"
    tables_path = data_dir / "tables.json"

    if not tables_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tables.json tại: {tables_path}")

    tables_data = load_json(tables_path)

    train_data: List[Dict[str, Any]] = []
    dev_data: List[Dict[str, Any]] = []

    if train_path.exists():
        train_data = load_json(train_path)
    else:
        print(f"[Cảnh báo] Không tìm thấy {train_path}. Vui lòng tải dataset ViText2SQL.")

    if dev_path.exists():
        dev_data = load_json(dev_path)
    else:
        print(f"[Cảnh báo] Không tìm thấy {dev_path}. Vui lòng tải dataset ViText2SQL.")

    return train_data, dev_data, tables_data


def load_test_gold_sql(test_gold_path: str | Path) -> List[Dict[str, str]]:
    """
    Đọc file test_gold.sql theo chuẩn Spider/ViText2SQL.

    Mỗi dòng: ``<gold SQL>\\t<db_id>``
    """
    test_gold_path = Path(test_gold_path)
    gold_records: List[Dict[str, str]] = []

    with open(test_gold_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Dòng không hợp lệ trong {test_gold_path}: cần định dạng 'SQL\\tdb_id'"
                )
            gold_records.append({"query": parts[0], "db_id": parts[1]})

    return gold_records


def load_eval_split(
    data_dir: str | Path,
    split: str = "test",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Tải tập dữ liệu dùng cho inference/đánh giá.

    Args:
        data_dir: Thư mục word-level hoặc syllable-level
        split: ``test`` (mặc định, theo paper) hoặc ``dev``

    Returns:
        (eval_data, train_data, tables_data)

    Tập **test**:
        - Câu hỏi lấy từ ``test.json``
        - SQL gốc lấy từ ``test.json`` (nếu có) hoặc ghép theo thứ tự từ ``test_gold.sql``
    """
    if split not in ("test", "dev"):
        raise ValueError(f"split phải là 'test' hoặc 'dev', nhận được: {split}")

    data_dir = Path(data_dir)
    train_data, dev_data, tables_data = load_vitext2sql_data(data_dir)

    if split == "dev":
        return dev_data, train_data, tables_data

    test_json_path = data_dir / "test.json"
    test_gold_path = data_dir / "test_gold.sql"

    if not test_json_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {test_json_path}. "
            "Tải test.json từ bộ ViText2SQL gốc (VinAIResearch)."
        )

    test_data = load_json(test_json_path)

    if test_gold_path.exists():
        gold_records = load_test_gold_sql(test_gold_path)
        if len(gold_records) != len(test_data):
            raise ValueError(
                f"Số mẫu không khớp: test.json ({len(test_data)}) "
                f"vs test_gold.sql ({len(gold_records)})"
            )
        for index, item in enumerate(test_data):
            # Luôn dùng gold SQL từ test_gold.sql (chuẩn đánh giá chính thức)
            item["query"] = gold_records[index]["query"]
            item["gold_query_toks"] = gold_records[index].get("query_toks")
            gold_db_id = gold_records[index]["db_id"]
            if item.get("db_id") and item["db_id"] != gold_db_id:
                print(
                    f"[Cảnh báo] Mẫu {index}: db_id test.json ({item['db_id']}) "
                    f"khác test_gold.sql ({gold_db_id}), giữ db_id từ test.json."
                )
            else:
                item["db_id"] = gold_db_id
    else:
        missing_query = [item for item in test_data if not item.get("query")]
        if missing_query:
            raise FileNotFoundError(
                f"{len(missing_query)} mẫu test thiếu trường 'query' "
                f"và không có {test_gold_path}."
            )

    return test_data, train_data, tables_data


def build_tables_index(tables_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Tạo index db_id -> mục schema trong tables.json."""
    return {entry["db_id"]: entry for entry in tables_data}


def format_create_table_statement(
    table_name: str,
    columns: List[Tuple[int, str]],
    column_types: List[str],
    column_name_offset: int,
) -> str:
    """Sinh lệnh CREATE TABLE cho một bảng (dùng trong prompt schema)."""
    column_definitions: List[str] = []
    for column_index, column_name in columns:
        type_index = column_name_offset + column_index
        column_type = column_types[type_index] if type_index < len(column_types) else "text"
        column_definitions.append(f'  "{column_name}" {column_type}')
    body = ",\n".join(column_definitions)
    return f'CREATE TABLE "{table_name}" (\n{body}\n);'


def build_full_schema_text(table_entry: Dict[str, Any]) -> str:
    """
    Xây dựng nội dung schema đầy đủ từ mục tables.json.

    Hiển thị tên bảng/cột tiếng Việt (table_names, column_names).
    """
    table_names = table_entry["table_names"]
    column_names = table_entry["column_names"]
    column_types = table_entry.get("column_types", [])

    statements: List[str] = []
    for table_index, table_name in enumerate(table_names):
        table_columns = [
            (column_index, column_name)
            for column_index, column_name in column_names
            if column_index == table_index
        ]
        # Tính offset cho column_types (column_names bao gồm cột * ở index -1)
        offset = sum(
            1 for column_index, _ in column_names if column_index < table_index
        )
        statements.append(
            format_create_table_statement(
                table_name, table_columns, column_types, offset
            )
        )

    foreign_keys = table_entry.get("foreign_keys", [])
    if foreign_keys:
        fk_lines: List[str] = []
        for source_index, target_index in foreign_keys:
            source_col = column_names[source_index][1]
            target_col = column_names[target_index][1]
            source_table = table_names[column_names[source_index][0]]
            target_table = table_names[column_names[target_index][0]]
            fk_lines.append(
                f'-- Khóa ngoại: "{source_table}"."{source_col}" -> '
                f'"{target_table}"."{target_col}"'
            )
        statements.append("\n".join(fk_lines))

    return "\n\n".join(statements)


def extract_sql_keywords(sql: str) -> set:
    """Trích xuất tên bảng/cột xuất hiện trong câu SQL (phục vụ schema linking)."""
    parsed = sqlparse.parse(sql)
    if not parsed:
        return set()
    tokens = set()
    for token in parsed[0].flatten():
        if token.ttype is None and not token.is_keyword and token.value.strip():
            value = token.value.strip().lower().replace('"', "").replace("`", "")
            if re.match(r"^[a-zA-Zà-ỹÀ-Ỹ0-9_\s]+$", value):
                tokens.add(value)
    return tokens


def schema_linking(
    table_entry: Dict[str, Any],
    question: str,
    reference_sql: Optional[str] = None,
    max_tables: Optional[int] = None,
) -> str:
    """
    Schema Linking: lọc schema, chỉ giữ bảng/cột liên quan đến câu hỏi.

    Chiến lược:
    1. Khớp tên bảng/cột xuất hiện trong câu hỏi (tiếng Việt)
    2. Nếu có reference_sql (few-shot), bổ sung bảng/cột từ SQL mẫu
    3. Luôn giữ ít nhất một bảng nếu không khớp gì (tránh schema rỗng)
    """
    table_names = table_entry["table_names"]
    column_names = table_entry["column_names"]
    column_types = table_entry.get("column_types", [])

    question_lower = question.lower()
    sql_keywords: set = set()
    if reference_sql:
        sql_keywords = extract_sql_keywords(reference_sql)

    relevant_table_indices: set = set()

    for table_index, table_name in enumerate(table_names):
        table_lower = table_name.lower()
        if table_lower in question_lower or table_lower in sql_keywords:
            relevant_table_indices.add(table_index)
            continue
        for column_index, column_name in column_names:
            if column_index == table_index and column_name.lower() in question_lower:
                relevant_table_indices.add(table_index)
                break

    if not relevant_table_indices:
        relevant_table_indices = set(range(len(table_names)))

    if max_tables and len(relevant_table_indices) > max_tables:
        relevant_table_indices = set(list(relevant_table_indices)[:max_tables])

    statements: List[str] = []
    for table_index in sorted(relevant_table_indices):
        table_name = table_names[table_index]
        table_columns = [
            (column_index, column_name)
            for column_index, column_name in column_names
            if column_index == table_index
        ]
        offset = sum(
            1 for column_index, _ in column_names if column_index < table_index
        )
        statements.append(
            format_create_table_statement(
                table_name, table_columns, column_types, offset
            )
        )

    return "\n\n".join(statements)


def format_few_shot_examples(examples: List[Dict[str, str]]) -> str:
    """Định dạng các ví dụ few-shot trong prompt."""
    blocks: List[str] = []
    for example in examples:
        blocks.append(
            f"Câu hỏi: {example['question']}\nSQL: {example['sql']}"
        )
    return "\n\n".join(blocks)


def build_text2sql_prompt(
    question: str,
    schema_content: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Xây dựng prompt Text-to-SQL theo định dạng yêu cầu (Việt + Anh).

    Args:
        question: Câu hỏi tiếng Việt
        schema_content: Nội dung schema (CREATE TABLE ...)
        few_shot_examples: Danh sách ví dụ [{"question": ..., "sql": ...}]
    """
    prompt_parts = [
        "Dựa vào lược đồ cơ sở dữ liệu (Database Schema) sau đây, "
        "hãy viết câu lệnh SQL chính xác cho câu hỏi tiếng Việt.",
        "",
        "[Schema]",
        schema_content,
    ]

    if few_shot_examples:
        prompt_parts.extend(["", "[Ví dụ]", format_few_shot_examples(few_shot_examples)])

    prompt_parts.extend([
        "",
        "[Bài tập]",
        f"Câu hỏi: {question}",
        "SQL:",
    ])

    return "\n".join(prompt_parts)


def tokenize_question_for_bm25(question: str) -> List[str]:
    """Tokenize câu hỏi tiếng Việt đơn giản cho BM25 (tách theo khoảng trắng)."""
    return question.lower().split()


def select_few_shot_examples_bm25(
    target_question: str,
    candidate_pool: List[Dict[str, Any]],
    num_examples: int = 3,
    same_database_only: bool = True,
    target_db_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Chọn ví dụ few-shot bằng BM25 (câu hỏi tương đồng nhất).

    Args:
        target_question: Câu hỏi cần dự đoán
        candidate_pool: Tập ứng viên (thường là train set)
        num_examples: Số ví dụ cần lấy
        same_database_only: Chỉ lấy ví dụ cùng database
        target_db_id: db_id của câu hỏi mục tiêu
    """
    filtered = candidate_pool
    if same_database_only and target_db_id:
        filtered = [item for item in candidate_pool if item.get("db_id") == target_db_id]

    if not filtered:
        filtered = candidate_pool

    corpus = [tokenize_question_for_bm25(item["question"]) for item in filtered]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize_question_for_bm25(target_question))

    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    examples: List[Dict[str, str]] = []
    for index in ranked_indices:
        item = filtered[index]
        if item["question"].strip() == target_question.strip():
            continue
        examples.append({"question": item["question"], "sql": item["query"]})
        if len(examples) >= num_examples:
            break

    return examples


def select_few_shot_examples_random(
    candidate_pool: List[Dict[str, Any]],
    num_examples: int = 3,
    same_database_only: bool = True,
    target_db_id: Optional[str] = None,
    exclude_question: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Chọn ví dụ few-shot ngẫu nhiên từ tập train."""
    if seed is not None:
        random.seed(seed)

    filtered = candidate_pool
    if same_database_only and target_db_id:
        filtered = [item for item in candidate_pool if item.get("db_id") == target_db_id]

    if exclude_question:
        filtered = [
            item for item in filtered
            if item["question"].strip() != exclude_question.strip()
        ]

    if len(filtered) < num_examples:
        filtered = candidate_pool

    sampled = random.sample(filtered, min(num_examples, len(filtered)))
    return [{"question": item["question"], "sql": item["query"]} for item in sampled]


def normalize_predicted_sql(raw_output: str) -> str:
    """
    Trích xuất và chuẩn hóa SQL từ output của mô hình.

    - Loại bỏ markdown code block
    - Lấy dòng bắt đầu bằng SELECT (hoặc WITH)
    - Chuẩn hóa khoảng trắng theo format ViText2SQL
    """
    text = raw_output.strip()

    # Trích xuất từ code block markdown
    code_block_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Tìm câu SQL (bắt đầu bằng select/with)
    lines = text.split("\n")
    sql_lines: List[str] = []
    capturing = False
    for line in lines:
        line_stripped = line.strip()
        if not capturing and re.match(r"^(select|with)\b", line_stripped, re.IGNORECASE):
            capturing = True
        if capturing:
            if line_stripped.lower().startswith("câu hỏi:"):
                break
            sql_lines.append(line_stripped)

    if sql_lines:
        text = " ".join(sql_lines)
    elif "SQL:" in text:
        text = text.split("SQL:")[-1].strip()

    # Loại bỏ khoảng trắng thừa, thêm khoảng trắng quanh toán tử
    text = text.lower().strip().rstrip(";")
    text = re.sub(r"([,()=<>!])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_database_path(db_id: str, database_dir: str | Path = DEFAULT_DATABASE_DIR) -> Path:
    """Trả về đường dẫn file SQLite cho db_id."""
    database_dir = Path(database_dir)
    return database_dir / db_id / f"{db_id}.sqlite"


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Tạo thư mục output nếu chưa tồn tại."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_model_config(model_key: str) -> Dict[str, Any]:
    """Lấy cấu hình mô hình từ MODEL_REGISTRY."""
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Mô hình '{model_key}' không được hỗ trợ. Có sẵn: {available}")
    return MODEL_REGISTRY[model_key]
