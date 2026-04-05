#!/bin/bash
# Full MT-Bench training + evaluation pipeline
# Model: Mistral-7B-v0.3, Dataset: oasst1, 8 GPUs
# Methods: SIFT, SpiEL, SMT, S2FT, LT-SFT
set -euo pipefail

RAPA_BASE="/home1/irteam/rapa"
LMFLOW_DIR="${RAPA_BASE}/LMFlow"
DATA="${RAPA_BASE}/data/oasst1_lmflow.json"
MODEL="mistralai/Mistral-7B-v0.3"
RESULTS="/home1/irteam/rapa/results/results.md"

source "${RAPA_BASE}/.rapa/bin/activate"
export HF_HOME="${RAPA_BASE}/hf_cache"
export HF_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY}"

mkdir -p "${RAPA_BASE}/results" /workspace/rapa

cd "${LMFLOW_DIR}"

for METHOD in sift spiel smt s2ft ltsft; do
    CKPT="${RAPA_BASE}/checkpoints/mtbench_${METHOD}"
    LOG="${CKPT}/logs"
    mkdir -p "${CKPT}" "${LOG}"

    echo "========================================"
    echo "[MT-Bench] Training: ${METHOD}"
    echo "========================================"

    PORT=$((29600 + RANDOM % 1000))
    deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${PORT} \
        examples/rapa/train_method.py \
        --method ${METHOD} \
        --model_name_or_path ${MODEL} \
        --dataset_path ${DATA} \
        --output_dir ${CKPT} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --lr_scheduler_type linear \
        --max_seq_length 512 \
        --target_params 170000000 \
        --bf16 \
        --hf_token ${HF_TOKEN} \
        --seed 42 \
        2>&1 | tee "${LOG}/train.log"

    TRAIN_EXIT=$?
    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "[MT-Bench] ${METHOD} training FAILED (exit ${TRAIN_EXIT})"
        continue
    fi
    echo "[MT-Bench] ${METHOD} training DONE"

    # Evaluate with MT-Bench using vLLM
    echo "[MT-Bench] Evaluating: ${METHOD}"
    python "${LMFLOW_DIR}/examples/rapa/eval_mtbench.py" \
        --model_path "${CKPT}" \
        --method "${METHOD}" \
        --results_file "${RESULTS}" \
        --judge "gpt-4o-mini" \
        --openai_api_key "${OPENAI_API_KEY}" \
        --num_gpus 8 \
        2>&1 | tee "${LOG}/eval.log"

    echo "[MT-Bench] ${METHOD} evaluation DONE"
done

echo "========================================"
echo "[MT-Bench] ALL METHODS COMPLETE"
echo "Results: ${RESULTS}"
echo "========================================"
