"""
Suy luận Zero-shot và Few-shot cho Text-to-SQL trên ViText2SQL.

Hỗ trợ:
- Zero-shot: không có ví dụ
- Few-shot: ví dụ ngẫu nhiên hoặc BM25
- Backend: HuggingFace Transformers (mặc định) hoặc vLLM (tùy chọn)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils import (
    build_tables_index,
    build_text2sql_prompt,
    ensure_output_dir,
    load_eval_split,
    normalize_predicted_sql,
    resolve_model_config,
    save_json,
    schema_linking,
    select_few_shot_examples_bm25,
    select_few_shot_examples_random,
)


class Text2SQLInferenceEngine:
    """Engine suy luận Text-to-SQL với HuggingFace Transformers."""

    def __init__(
        self,
        model_key: str,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        use_unsloth: bool = False,
    ):
        """
        Khởi tạo engine suy luận.

        Args:
            model_key: Khóa mô hình trong MODEL_REGISTRY
            device: Thiết bị ('cuda', 'cpu', 'auto')
            load_in_4bit: Quantization 4-bit để tiết kiệm VRAM
            use_unsloth: Dùng Unsloth FastLanguageModel (nhanh hơn, ít VRAM hơn)
        """
        self.model_config = resolve_model_config(model_key)
        self.model_name = self.model_config["model_name"]
        self.max_seq_length = self.model_config.get("max_seq_length", 4096)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_unsloth = use_unsloth

        if use_unsloth:
            self._load_unsloth_model(load_in_4bit)
        else:
            self._load_hf_model(load_in_4bit)

    def _load_unsloth_model(self, load_in_4bit: bool) -> None:
        """Tải mô hình qua Unsloth FastLanguageModel."""
        try:
            from unsloth import FastLanguageModel
        except ImportError as error:
            raise ImportError(
                "Cần cài unsloth: pip install unsloth. "
                "Hoặc dùng --no_unsloth để suy luận bằng HuggingFace thuần."
            ) from error

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)

    def _load_hf_model(self, load_in_4bit: bool) -> None:
        """Tải mô hình qua HuggingFace Transformers."""
        quantization_config = None
        if load_in_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()

    def generate_sql(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> str:
        """Sinh câu SQL từ prompt."""
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                input_text = prompt
        else:
            input_text = prompt

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length - max_new_tokens,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return normalize_predicted_sql(generated)

    def generate_sql_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> List[str]:
        """Sinh nhiều SQL cùng lúc từ một danh sách prompt.

        Lưu ý: với `use_unsloth` sẽ fallback về vòng lặp đơn.
        """
        if self.use_unsloth:
            return [self.generate_sql(p, max_new_tokens, temperature, top_p) for p in prompts]

        # Áp dụng template chat nếu tokenizer hỗ trợ
        input_texts: List[str] = []
        if hasattr(self.tokenizer, "apply_chat_template"):
            for prompt in prompts:
                try:
                    input_text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    input_text = prompt
                input_texts.append(input_text)
        else:
            input_texts = prompts

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length - max_new_tokens,
        )

        # Lưu độ dài input thực tế để tách phần sinh ra sau này
        input_lengths = inputs["attention_mask"].sum(dim=1)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        results: List[str] = []
        for i, input_len in enumerate(input_lengths):
            gen_tokens = outputs[i, input_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            results.append(normalize_predicted_sql(text))

        return results


def run_inference(
    split: str,
    model_key: str,
    mode: str,
    output_path: Path,
    data_dir: Path,
    num_shots: int = 3,
    example_strategy: str = "bm25",
    max_samples: Optional[int] = None,
    load_in_4bit: bool = True,
    use_unsloth: bool = False,
    seed: int = 42,
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """
    Chạy suy luận trên tập dev hoặc test.

    Args:
        split: 'dev' hoặc 'test'
        model_key: Khóa mô hình
        mode: 'zero_shot' hoặc 'few_shot'
        output_path: Đường dẫn lưu prediction.json
        data_dir: Thư mục dữ liệu
        num_shots: Số ví dụ few-shot
        example_strategy: 'bm25' hoặc 'random'
        max_samples: Giới hạn số mẫu (debug)
    """
    eval_data, train_data, tables_data = load_eval_split(data_dir, split=split)
    tables_index = build_tables_index(tables_data)

    if max_samples:
        eval_data = eval_data[:max_samples]

    engine = Text2SQLInferenceEngine(
        model_key=model_key,
        load_in_4bit=load_in_4bit,
        use_unsloth=use_unsloth,
    )

    predictions: List[Dict[str, Any]] = []

    # Xử lý theo batch
    for start in tqdm(range(0, len(eval_data), batch_size), desc=f"{mode} - {model_key} - {split}"):
        batch_items = eval_data[start : start + batch_size]
        prompts: List[str] = []
        per_item_meta: List[Dict[str, Any]] = []

        for item in batch_items:
            db_id = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]

            table_entry = tables_index[db_id]

            few_shot_examples = None
            if mode == "few_shot":
                if example_strategy == "bm25":
                    few_shot_examples = select_few_shot_examples_bm25(
                        target_question=question,
                        candidate_pool=train_data,
                        num_examples=num_shots,
                        target_db_id=db_id,
                    )
                else:
                    few_shot_examples = select_few_shot_examples_random(
                        candidate_pool=train_data,
                        num_examples=num_shots,
                        target_db_id=db_id,
                        exclude_question=question,
                        seed=seed,
                    )

            reference_sql = few_shot_examples[0]["sql"] if few_shot_examples else None
            schema_text = schema_linking(
                table_entry=table_entry,
                question=question,
                reference_sql=reference_sql,
            )

            prompt = build_text2sql_prompt(
                question=question,
                schema_content=schema_text,
                few_shot_examples=few_shot_examples,
            )

            prompts.append(prompt)
            per_item_meta.append({
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "gold_query_toks": item.get("query_toks"),
            })

        # Sinh theo batch (engine sẽ fallback sang vòng lặp nếu không hỗ trợ)
        if len(prompts) == 1:
            predicted_sqls = [engine.generate_sql(prompts[0])]
        else:
            predicted_sqls = engine.generate_sql_batch(prompts)

        for meta, predicted_sql in zip(per_item_meta, predicted_sqls):
            predictions.append({
                "db_id": meta["db_id"],
                "question": meta["question"],
                "gold_sql": meta["gold_sql"],
                "predict_sql": predicted_sql,
                "gold_query_toks": meta.get("gold_query_toks"),
                "mode": mode,
                "model": model_key,
            })

    save_json(predictions, output_path)
    print(f"Đã lưu {len(predictions)} dự đoán tại: {output_path}")
    return predictions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suy luận Zero-shot / Few-shot Text-to-SQL trên ViText2SQL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder-1.5b",
        choices=["qwen2.5-coder-1.5b", "llama-3.2-3b", "gemma-2-2b"],
        help="Mô hình LLM phân khúc ~2B",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Chế độ suy luận",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="Tập dữ liệu đánh giá",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/word-level",
        help="Thư mục chứa train.json, dev.json, tables.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Thư mục lưu kết quả",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=3,
        help="Số ví dụ few-shot",
    )
    parser.add_argument(
        "--example_strategy",
        type=str,
        default="bm25",
        choices=["bm25", "random"],
        help="Chiến lược chọn ví dụ few-shot",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu (debug)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Kích thước batch để suy luận (mặc định 1 = không batch).",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Tắt quantization 4-bit",
    )
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        help="Dùng Unsloth FastLanguageModel cho suy luận",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = ensure_output_dir(args.output_dir)
    output_file = output_dir / f"predictions_{args.model}_{args.mode}_{args.split}.json"

    run_inference(
        split=args.split,
        model_key=args.model,
        mode=args.mode,
        output_path=output_file,
        data_dir=Path(args.data_dir),
        num_shots=args.num_shots,
        example_strategy=args.example_strategy,
        max_samples=args.max_samples,
        load_in_4bit=not args.no_4bit,
        use_unsloth=args.use_unsloth,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
