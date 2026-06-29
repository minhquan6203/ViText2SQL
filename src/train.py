"""
Fine-tune LLM với QLoRA (Unsloth) trên bộ dữ liệu ViText2SQL.

Tối ưu cho GPU cấu hình thấp (T4 16GB, A10G 24GB):
- 4-bit quantization (QLoRA)
- FastLanguageModel từ Unsloth (tiết kiệm ~60% VRAM)
- LoRA rank=16, alpha=32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset
from tqdm import tqdm

from src.inference import Text2SQLInferenceEngine
from src.utils import (
    build_tables_index,
    build_text2sql_prompt,
    ensure_output_dir,
    load_vitext2sql_data,
    resolve_model_config,
    save_json,
    schema_linking,
)


# Cấu hình QLoRA theo yêu cầu
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_DROPOUT = 0.05


def build_training_dataset(
    train_data: List[Dict[str, Any]],
    tables_index: Dict[str, Dict[str, Any]],
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Chuyển train.json thành dataset huấn luyện instruction-following.

    Format mỗi mẫu:
        instruction: prompt (schema + câu hỏi)
        output: câu SQL gốc
    """
    records: List[Dict[str, str]] = []

    data_subset = train_data[:max_samples] if max_samples else train_data

    for item in data_subset:
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        table_entry = tables_index[db_id]

        schema_text = schema_linking(table_entry, question, reference_sql=gold_sql)
        prompt = build_text2sql_prompt(question=question, schema_content=schema_text)

        records.append({
            "instruction": prompt,
            "output": gold_sql,
            "db_id": db_id,
            "question": question,
        })

    return Dataset.from_list(records)


def format_training_text(example: Dict[str, str], tokenizer) -> Dict[str, str]:
    """
    Định dạng mẫu huấn luyện theo chat template của mô hình.

    Mô hình học sinh SQL sau phần 'SQL:' trong prompt.
    """
    user_content = example["instruction"]
    assistant_content = example["output"]

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = f"{user_content}\n{assistant_content}"

    return {"text": text}


def train_qlora(
    model_key: str,
    data_dir: Path,
    output_dir: Path,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 4096,
    max_train_samples: Optional[int] = None,
    eval_after_train: bool = True,
    eval_split: str = "dev",
) -> Path:
    """
    Huấn luyện QLoRA với Unsloth và lưu checkpoint.

    Returns:
        Đường dẫn thư mục checkpoint
    """
    try:
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise ImportError(
            "Cần cài unsloth và trl: pip install unsloth trl transformers"
        ) from error

    model_config = resolve_model_config(model_key)
    model_name = model_config["model_name"]

    train_data, dev_data, tables_data = load_vitext2sql_data(data_dir)
    tables_index = build_tables_index(tables_data)

    print(f"[1/5] Tải mô hình {model_name} với Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"[2/5] Gắn adapter LoRA (r={LORA_RANK}, alpha={LORA_ALPHA})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("[3/5] Chuẩn bị dataset huấn luyện...")
    raw_dataset = build_training_dataset(
        train_data, tables_index, max_samples=max_train_samples
    )
    formatted_dataset = raw_dataset.map(
        lambda example: format_training_text(example, tokenizer),
        remove_columns=raw_dataset.column_names,
    )

    checkpoint_dir = output_dir / f"checkpoints_{model_key}_qlora"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = batch_size * gradient_accumulation_steps
    steps_per_epoch = max(
        1, (len(formatted_dataset) + effective_batch - 1) // effective_batch
    )
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    # Dùng SFTConfig (không dùng TrainingArguments) để tránh lỗi pickle
    # khi Unsloth lưu checkpoint: "Can't pickle SFTConfig: not the same object".
    training_arguments = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=42,
        max_length=max_seq_length,
        dataset_text_field="text",
    )

    print("[4/5] Bắt đầu huấn luyện QLoRA...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        args=training_arguments,
    )
    trainer.train()

    # Lưu adapter LoRA và tokenizer
    final_checkpoint = checkpoint_dir / "final"
    model.save_pretrained(str(final_checkpoint))
    tokenizer.save_pretrained(str(final_checkpoint))
    print(f"Đã lưu checkpoint tại: {final_checkpoint}")

    if eval_after_train:
        print("[5/5] Sinh prediction.json sau huấn luyện...")
        generate_post_train_predictions(
            model_key=model_key,
            checkpoint_path=final_checkpoint,
            dev_data=dev_data if eval_split == "dev" else [],
            tables_index=tables_index,
            output_dir=output_dir,
            eval_split=eval_split,
            data_dir=data_dir,
        )
    else:
        print("[5/5] Bỏ qua đánh giá sau train.")

    return final_checkpoint


def generate_post_train_predictions(
    model_key: str,
    checkpoint_path: Path,
    dev_data: List[Dict[str, Any]],
    tables_index: Dict[str, Dict[str, Any]],
    output_dir: Path,
    eval_split: str,
    data_dir: Path,
    max_samples: Optional[int] = None,
) -> Path:
    """
    Sinh file prediction.json sau fine-tune.

    File chứa: [Câu hỏi, SQL gốc, SQL dự đoán] cho từng mẫu.
    """
    if eval_split == "dev":
        _, eval_data, _ = load_vitext2sql_data(data_dir)
    else:
        test_path = data_dir / "test.json"
        with open(test_path, "r", encoding="utf-8") as file:
            eval_data = json.load(file)

    if max_samples:
        eval_data = eval_data[:max_samples]

    # Tải mô hình đã fine-tune qua Unsloth
    try:
        from unsloth import FastLanguageModel
    except ImportError as error:
        raise ImportError("Cần unsloth để load checkpoint.") from error

    model_config = resolve_model_config(model_key)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_path),
        max_seq_length=model_config.get("max_seq_length", 4096),
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Tạo engine tạm với mô hình đã load
    engine = Text2SQLInferenceEngine.__new__(Text2SQLInferenceEngine)
    engine.model = model
    engine.tokenizer = tokenizer
    engine.max_seq_length = model_config.get("max_seq_length", 4096)
    engine.model_config = model_config

    predictions: List[Dict[str, Any]] = []

    for item in tqdm(eval_data, desc="Suy luận sau fine-tune"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        table_entry = tables_index[db_id]

        schema_text = schema_linking(table_entry, question, reference_sql=gold_sql)
        prompt = build_text2sql_prompt(question=question, schema_content=schema_text)
        predicted_sql = engine.generate_sql(prompt)

        predictions.append({
            "question": question,
            "gold_sql": gold_sql,
            "predict_sql": predicted_sql,
            "db_id": db_id,
            "gold_query_toks": item.get("query_toks"),
            "mode": "fine_tuned",
            "model": model_key,
        })

    output_file = output_dir / f"predictions_{model_key}_finetuned_{eval_split}.json"
    save_json(predictions, output_file)
    print(f"Đã lưu prediction.json tại: {output_file}")
    return output_file


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune QLoRA với Unsloth trên ViText2SQL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder-1.5b",
        choices=["qwen2.5-coder-1.5b", "llama-3.2-3b", "gemma-2-2b"],
    )
    parser.add_argument("--data_dir", type=str, default="data/syllable-level")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--no_eval_after_train", action="store_true")
    parser.add_argument("--eval_split", type=str, default="dev", choices=["dev", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = ensure_output_dir(args.output_dir)

    train_qlora(
        model_key=args.model,
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        max_train_samples=args.max_train_samples,
        eval_after_train=not args.no_eval_after_train,
        eval_split=args.eval_split,
    )


if __name__ == "__main__":
    main()
