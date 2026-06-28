#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh - Pipeline tự động ViText2SQL
#
# Luồng thực thi:
#   1. Đánh giá Zero-shot trên dev set
#   2. Đánh giá Few-shot (BM25) trên dev set
#   3. Fine-tune QLoRA với Unsloth
#   4. Đánh giá sau fine-tune
#
# Cách dùng:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                          # Chạy toàn bộ với Qwen 1.5B
#   ./run_pipeline.sh --model llama-3.2-3b     # Chọn mô hình khác
#   ./run_pipeline.sh --skip_train             # Chỉ inference + eval
# =============================================================================

set -euo pipefail

# --- Cấu hình mặc định ---
MODEL="${MODEL:-qwen2.5-coder-1.5b}"
DATA_DIR="${DATA_DIR:-data/word-level}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
SPLIT="${SPLIT:-test}"
NUM_SHOTS="${NUM_SHOTS:-3}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
USE_UNSLOTH="${USE_UNSLOTH:-1}"

# --- Parse tham số dòng lệnh ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num_shots)
            NUM_SHOTS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --skip_train)
            SKIP_TRAIN=1
            shift
            ;;
        --no_unsloth)
            USE_UNSLOTH=0
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Mô hình: qwen2.5-coder-1.5b | llama-3.2-3b | gemma-2-2b"
            echo "  --data_dir PATH        Thư mục dữ liệu (mặc định: data/word-level)"
            echo "  --output_dir PATH      Thư mục output (mặc định: outputs)"
            echo "  --split SPLIT          dev hoặc test"
            echo "  --num_shots N          Số ví dụ few-shot (mặc định: 3)"
            echo "  --num_epochs N         Số epoch fine-tune (mặc định: 3)"
            echo "  --max_samples N        Giới hạn mẫu (debug)"
            echo "  --skip_train           Bỏ qua bước fine-tune"
            echo "  --no_unsloth           Không dùng Unsloth cho inference"
            exit 0
            ;;
        *)
            echo "Tham số không hợp lệ: $1"
            exit 1
            ;;
    esac
done

# --- Thiết lập môi trường ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Tham số tùy chọn max_samples
MAX_SAMPLES_ARG=""
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

UNSLOTH_ARG=""
if [[ "$USE_UNSLOTH" == "1" ]]; then
    UNSLOTH_ARG="--use_unsloth"
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  ViText2SQL Pipeline"
echo "  Mô hình:    $MODEL"
echo "  Dữ liệu:    $DATA_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Split:      $SPLIT"
echo "=============================================="

# --- Kiểm tra dữ liệu ---
if [[ ! -f "$DATA_DIR/tables.json" ]]; then
    echo "[LỖI] Không tìm thấy $DATA_DIR/tables.json"
    echo "      Vui lòng tải dataset ViText2SQL từ VinAIResearch."
    exit 1
fi

if [[ ! -f "$DATA_DIR/dev.json" ]]; then
    echo "[CẢNH BÁO] Không tìm thấy dev.json. Một số bước có thể thất bại."
fi

# =============================================================================
# BƯỚC 1: Zero-shot Inference + Evaluation
# =============================================================================
echo ""
echo ">>> [Bước 1/4] Zero-shot Inference..."
python -m src.inference \
    --model "$MODEL" \
    --mode zero_shot \
    --split "$SPLIT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $MAX_SAMPLES_ARG \
    $UNSLOTH_ARG

PRED_ZERO="$OUTPUT_DIR/predictions_${MODEL}_zero_shot_${SPLIT}.json"

echo ">>> [Bước 1/4] Đánh giá Zero-shot..."
python -m src.evaluate \
    --predictions "$PRED_ZERO" \
    --tables "$DATA_DIR/tables.json" \
    --output_metrics "$OUTPUT_DIR/metrics_${MODEL}_zero_shot_${SPLIT}.json"

# =============================================================================
# BƯỚC 2: Few-shot Inference (BM25) + Evaluation
# =============================================================================
echo ""
echo ">>> [Bước 2/4] Few-shot Inference (BM25)..."
python -m src.inference \
    --model "$MODEL" \
    --mode few_shot \
    --split "$SPLIT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_shots "$NUM_SHOTS" \
    --example_strategy bm25 \
    $MAX_SAMPLES_ARG \
    $UNSLOTH_ARG

PRED_FEW="$OUTPUT_DIR/predictions_${MODEL}_few_shot_${SPLIT}.json"

echo ">>> [Bước 2/4] Đánh giá Few-shot..."
python -m src.evaluate \
    --predictions "$PRED_FEW" \
    --tables "$DATA_DIR/tables.json" \
    --output_metrics "$OUTPUT_DIR/metrics_${MODEL}_few_shot_${SPLIT}.json"

# =============================================================================
# BƯỚC 3: Fine-tune QLoRA (Unsloth)
# =============================================================================
if [[ "$SKIP_TRAIN" == "0" ]]; then
    echo ""
    echo ">>> [Bước 3/4] Fine-tune QLoRA với Unsloth..."
    python -m src.train \
        --model "$MODEL" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --eval_split "$SPLIT" \
        $MAX_SAMPLES_ARG

    # =============================================================================
    # BƯỚC 4: Đánh giá sau Fine-tune
    # =============================================================================
    echo ""
    echo ">>> [Bước 4/4] Đánh giá sau Fine-tune..."
    PRED_FT="$OUTPUT_DIR/predictions_${MODEL}_finetuned_${SPLIT}.json"

    python -m src.evaluate \
        --predictions "$PRED_FT" \
        --tables "$DATA_DIR/tables.json" \
        --output_metrics "$OUTPUT_DIR/metrics_${MODEL}_finetuned_${SPLIT}.json"
else
    echo ""
    echo ">>> [Bước 3-4] Bỏ qua fine-tune (--skip_train)"
fi

# =============================================================================
# Tổng kết
# =============================================================================
echo ""
echo "=============================================="
echo "  Pipeline hoàn tất!"
echo "  Kết quả lưu tại: $OUTPUT_DIR"
echo ""
echo "  Files chính:"
echo "    - predictions_${MODEL}_zero_shot_${SPLIT}.json"
echo "    - predictions_${MODEL}_few_shot_${SPLIT}.json"
if [[ "$SKIP_TRAIN" == "0" ]]; then
    echo "    - predictions_${MODEL}_finetuned_${SPLIT}.json"
    echo "    - checkpoints_${MODEL}_qlora/final/"
fi
echo "=============================================="
