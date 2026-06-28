# ViText2SQL — Pipeline Text-to-SQL cho tiếng Việt

Repository Python mô-đun để thực nghiệm **Text-to-SQL** trên bộ dữ liệu [ViText2SQL](https://github.com/VinAIResearch/ViText2SQL) (VinAI Research, EMNLP 2020 Findings).

Hỗ trợ đầy đủ ba hướng tiếp cận:

| Phương pháp | Mô tả |
|-------------|--------|
| **Zero-shot** | Schema + câu hỏi, không có ví dụ |
| **Few-shot** | Thêm ví dụ từ train set (BM25 hoặc random) |
| **Fine-tune (QLoRA)** | Huấn luyện adapter LoRA với Unsloth trên GPU cấu hình thấp |

Đánh giá theo **Exact Match (EM)** và **Component F1** kế thừa từ [Spider Benchmark](https://github.com/taoyds/spider) — so khớp cấu trúc AST SQL, không phải so sánh chuỗi.

---

## Mục lục

- [Cấu trúc repository](#cấu-trúc-repository)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Chạy pipeline tự động](#chạy-pipeline-tự-động)
- [Hướng dẫn từng module](#hướng-dẫn-từng-module)
- [Notebook API (DeepSeek, Gemini)](#notebook-api-deepseek-gemini)
- [Mô hình hỗ trợ](#mô-hình-hỗ-trợ)
- [Đánh giá (EM & F1)](#đánh-giá-em--f1)
- [Trích dẫn](#trích-dẫn)

---

## Cấu trúc repository

```
ViText2SQL/
├── data/
│   ├── word-level/              # Bản tách từ (RDRSegmenter)
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json            # Câu hỏi test
│   │   ├── test_gold.sql        # SQL gốc test (đánh giá chính thức)
│   │   └── tables.json
│   ├── syllable-level/          # Bản gốc mức âm tiết
│   └── database/                # SQLite DBs (tùy chọn, cho execution eval)
│       └── {db_id}/{db_id}.sqlite
├── notebooks/
│   └── api_zero_few_shot.ipynb  # Zero/Few-shot qua API (DeepSeek, Gemini)
├── outputs/                     # Predictions, metrics, checkpoints
├── src/
│   ├── utils.py                 # Tải dữ liệu, schema linking, prompt
│   ├── inference.py             # Zero-shot / Few-shot (local LLM)
│   ├── train.py                 # Fine-tune QLoRA (Unsloth)
│   ├── evaluate.py              # EM & Component F1
│   ├── api_clients.py           # Client DeepSeek, Gemini
│   └── spider_eval/             # Parser & logic đánh giá Spider
├── requirements.txt
└── run_pipeline.sh              # Script chạy toàn bộ luồng
```

---

## Yêu cầu hệ thống

| Thành phần | Khuyến nghị |
|------------|-------------|
| Python | 3.10+ |
| GPU (local LLM / fine-tune) | NVIDIA T4 16GB trở lên (QLoRA 4-bit) |
| RAM | ≥ 16 GB |
| API (DeepSeek/Gemini) | Không cần GPU |

---

## Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd ViText2SQL

# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Cài dependencies
pip install -r requirements.txt

# NLTK (cho tokenizer Spider gốc, nếu cần)
python -c "import nltk; nltk.download('punkt')"

# Thiết lập PYTHONPATH (hoặc chạy từ thư mục gốc)
export PYTHONPATH=.               # Linux/macOS
# $env:PYTHONPATH="."            # Windows PowerShell
```

> **Lưu ý:** Package `unsloth` yêu cầu GPU NVIDIA. Nếu chỉ dùng API hoặc inference HuggingFace thuần, có thể bỏ qua lỗi unsloth hoặc cài riêng khi cần fine-tune.

---

## Chuẩn bị dữ liệu

### Hai biến thể dữ liệu

| Biến thể | Thư mục | Mô tả |
|----------|---------|--------|
| **word-level** | `data/word-level/` | Câu hỏi đã tách từ |
| **syllable-level** | `data/syllable-level/` | Bản gốc mức âm tiết |

Chọn biến thể bằng `--data_dir data/word-level` hoặc `--data_dir data/syllable-level`.

### Tập đánh giá

| Split | File | Mục đích |
|-------|------|----------|
| **test** (mặc định) | `test.json` + `test_gold.sql` | Báo cáo kết quả chính thức (theo paper) |
| **dev** | `dev.json` | Tuning / debug nhanh |

Logic tải test: câu hỏi từ `test.json`, SQL gốc ghép từ `test_gold.sql` theo đúng thứ tự dòng (chuẩn Spider).

### Database SQLite (tùy chọn)

Đặt file DB theo cấu trúc Spider:

```
data/database/{db_id}/{db_id}.sqlite
```

Cần cho **Execution Accuracy**; EM/F1 structural matching chỉ cần `tables.json`.

---

## Chạy pipeline tự động

Script `run_pipeline.sh` chạy lần lượt: Zero-shot → Few-shot → Fine-tune → Eval trên **test set**.

```bash
chmod +x run_pipeline.sh

# Chạy toàn bộ (Qwen 1.5B, word-level, test set)
./run_pipeline.sh

# Tùy chọn
./run_pipeline.sh \
  --model qwen2.5-coder-1.5b \
  --data_dir data/syllable-level \
  --split test \
  --num_shots 3 \
  --num_epochs 3 \
  --max_samples 50          # Giới hạn mẫu (debug)

# Chỉ inference + eval, bỏ fine-tune
./run_pipeline.sh --skip_train
```

**Windows (Git Bash / WSL):** dùng lệnh trên trong Git Bash hoặc WSL. Trên PowerShell thuần, chạy từng lệnh Python bên dưới.

---

## Hướng dẫn từng module

### 1. Inference — Zero-shot / Few-shot (local LLM)

```bash
# Zero-shot trên test set
python -m src.inference \
  --model qwen2.5-coder-1.5b \
  --mode zero_shot \
  --split test \
  --data_dir data/word-level \
  --output_dir outputs

# Few-shot với BM25 (3 ví dụ từ train set)
python -m src.inference \
  --model llama-3.2-3b \
  --mode few_shot \
  --example_strategy bm25 \
  --num_shots 3 \
  --split test \
  --use_unsloth

# Few-shot random
python -m src.inference \
  --model gemma-2-2b \
  --mode few_shot \
  --example_strategy random \
  --split test
```

Kết quả: `outputs/predictions_{model}_{mode}_{split}.json`

### 2. Fine-tune QLoRA (Unsloth)

```bash
python -m src.train \
  --model qwen2.5-coder-1.5b \
  --data_dir data/word-level \
  --output_dir outputs \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --eval_split test
```

Cấu hình LoRA mặc định: `r=16`, `alpha=32`, target modules Q/K/V/O + MLP.

Checkpoint: `outputs/checkpoints_{model}_qlora/final/`

Sau train tự động sinh `predictions_{model}_finetuned_test.json`.

### 3. Đánh giá EM & F1

```bash
python -m src.evaluate \
  --predictions outputs/predictions_qwen2.5-coder-1.5b_zero_shot_test.json \
  --tables data/word-level/tables.json \
  --output_metrics outputs/metrics_zero_shot.json \
  --verbose
```

Xuất định dạng Spider (tab-separated):

```bash
python -m src.evaluate \
  --predictions outputs/predictions_....json \
  --tables data/word-level/tables.json \
  --export_spider_format outputs/spider_format
```

---

## Notebook API (DeepSeek, Gemini)

Dùng khi **không có GPU** hoặc muốn thử mô hình lớn qua API.

**File:** [`notebooks/api_zero_few_shot.ipynb`](notebooks/api_zero_few_shot.ipynb)

### Thiết lập API key

```bash
# DeepSeek — https://platform.deepseek.com/
export DEEPSEEK_API_KEY="sk-..."

# Google Gemini — https://aistudio.google.com/apikey
export GEMINI_API_KEY="..."
```

### Cấu hình trong notebook

```python
MODEL_KEY = "deepseek-coder"    # hoặc gemini-2.0-flash, gemini-1.5-pro
MODE = "few_shot"               # zero_shot | few_shot
SPLIT = "test"                  # test (mặc định) | dev
DATA_DIR = PROJECT_ROOT / "data" / "word-level"
MAX_SAMPLES = None              # None = chạy hết test set
```

Mở notebook → chạy tuần tự các cell → xem EM/F1 ở cuối.

---

## Mô hình hỗ trợ

### Local (HuggingFace + Unsloth)

| Khóa | Model ID |
|------|----------|
| `qwen2.5-coder-1.5b` | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| `llama-3.2-3b` | meta-llama/Llama-3.2-3B-Instruct |
| `gemma-2-2b` | google/gemma-2-2b-it |

### API

| Khóa | Provider |
|------|----------|
| `deepseek-chat` | DeepSeek |
| `deepseek-coder` | DeepSeek (khuyến nghị cho SQL) |
| `gemini-2.0-flash` | Google Gemini |
| `gemini-1.5-pro` | Google Gemini |
| `gemini-1.5-flash` | Google Gemini |

---

## Đánh giá (EM & F1)

### Exact Match (EM)

`1` nếu cây cú pháp SQL (AST) dự đoán **khớp hoàn toàn** với gold — gồm SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, INTERSECT/UNION/EXCEPT.

### Component F1

Tính Precision / Recall / F1 **độc lập từng thành phần** (không phạt sai thứ tự cột trong SELECT hay thứ tự điều kiện AND/OR):

- SELECT (có/không aggregation)
- WHERE
- GROUP BY (+ HAVING)
- ORDER BY (+ LIMIT)

Implementation: [`src/spider_eval/evaluation_core.py`](src/spider_eval/evaluation_core.py) — kế thừa [`spider/evaluation.py`](https://github.com/taoyds/spider/blob/master/evaluation.py).

---

## Định dạng Prompt

```
Dựa vào lược đồ cơ sở dữ liệu (Database Schema) sau đây, hãy viết câu lệnh SQL chính xác cho câu hỏi tiếng Việt.

[Schema]
CREATE TABLE "..." (...);

[Ví dụ]          ← chỉ khi few-shot
Câu hỏi: ...
SQL: ...

[Bài tập]
Câu hỏi: {câu hỏi tiếng Việt}
SQL:
```

---

## Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `ModuleNotFoundError: src` | Chạy từ thư mục gốc, đặt `PYTHONPATH=.` |
| OOM khi train | Giảm `--batch_size`, tăng `--gradient_accumulation_steps` |
| Unsloth không cài được | Dùng inference không `--use_unsloth`; train cần GPU NVIDIA |
| Test số mẫu không khớp | Kiểm tra `test.json` và `test_gold.sql` cùng thứ tự, cùng biến thể (word/syllable) |
| API rate limit | Tăng `time.sleep` trong notebook; giảm `MAX_SAMPLES` |

---

## Trích dẫn

### Dataset ViText2SQL (VinAI Research)

```bibtex
@inproceedings{vitext2sql,
    title     = {{A Pilot Study of Text-to-SQL Semantic Parsing for Vietnamese}},
    author    = {Anh Tuan Nguyen and Mai Hoang Dao and Dat Quoc Nguyen},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
    year      = {2020},
    pages     = {4079--4085}
}
```

### Spider Benchmark (metric đánh giá)

```bibtex
@inproceedings{Yu2018Spider,
    title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
    author    = {Tao Yu and Rui Zhang and Kai Yang and others},
    booktitle = {EMNLP},
    year      = {2018}
}
```

---

## Giấy phép dữ liệu

Bằng việc sử dụng ViText2SQL, bạn đồng ý:

- Chỉ dùng cho mục đích nghiên cứu hoặc giáo dục.
- **Không** phân phối ViText2SQL hoặc bất kỳ phần nào của bộ dữ liệu.
- Trích dẫn paper EMNLP 2020 Findings ở trên khi công bố kết quả.

#### Copyright (c) 2020 VinAI Research

```
THE DATA IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE DATA OR THE USE OR OTHER DEALINGS IN THE
DATA.
```