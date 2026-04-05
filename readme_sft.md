# Sparse Fine-Tuning Methods — Unified Benchmark

5가지 Sparse Fine-Tuning 방법을 LMFlow 기반으로 통합 구현하여 MT-Bench, MMLU, CSR 벤치마크에서 비교 평가합니다.

## Methods

| Method | Paper/Repo | Sparsity Type | Description |
|--------|-----------|---------------|-------------|
| **SIFT** | [song-wx/SIFT](https://github.com/song-wx/SIFT) | Random element-wise | 랜덤 위치의 weight element를 학습. `SparseLinear` 모듈로 frozen weight에 sparse delta 추가 |
| **SpiEL** | [ducdauge/sft-llm](https://github.com/ducdauge/sft-llm) + [AlanAnsell/peft](https://github.com/AlanAnsell/peft) | Scatter sparse | PEFT fork의 `SftConfig` 사용. Scatter 방식 sparse fine-tuning |
| **SMT** | [HectorHHZ/Sparse_Matrix_Tuning](https://github.com/HectorHHZ/Sparse_Matrix_Tuning) | Block-sparse (256x256) | 256x256 block 단위로 랜덤 선택하여 학습 |
| **S2FT** | [Infini-AI-Lab/S2FT](https://github.com/Infini-AI-Lab/S2FT) | Structured row | Weight matrix의 연속 row를 선택하여 학습 (structured sparsity) |
| **LT-SFT** | [cambridgeltl/composable-sft](https://github.com/cambridgeltl/composable-sft) | Lottery ticket | Lottery Ticket 방식 랜덤 element 선택. SIFT와 유사하나 별도 seed |

모든 method에서 **~170M trainable parameters** (전체 7B 모델의 ~2.35%)를 사용합니다.

## Architecture

```
sparse_ft/
├── methods/           # 각 method의 원본 코드 (수정 포함)
│   ├── sift/         # SIFT - SparseLinear + index cache
│   ├── spiel/        # SpiEL - AlanAnsell peft fork
│   ├── smt/          # SMT - BlockSparseLinear
│   ├── s2ft/         # S2FT - RowSparseLinear  
│   └── ltsft/        # LT-SFT - SparseLinear (lottery ticket)
├── pipeline/          # LMFlow 통합 파이프라인
│   ├── __init__.py
│   ├── sift_tuner.py  # SIFT: frozen weight(buffer) + sparse_delta(Parameter)
│   ├── spiel_tuner.py # SpiEL: peft fork SftConfig + merge_and_unload
│   ├── smt_tuner.py   # SMT: BlockSparseLinear (256x256 blocks)
│   ├── s2ft_tuner.py  # S2FT: RowSparseLinear (contiguous rows)
│   ├── ltsft_tuner.py # LT-SFT: SparseLinear (random elements)
│   ├── train_method.py      # 통합 학습 entrypoint
│   ├── eval_mtbench.py      # MT-Bench 평가 (vLLM + GPT-4o-mini judge)
│   ├── eval_lmharness.py    # MMLU/CSR 평가 (lm-eval-harness + vLLM)
│   ├── prepare_datasets.py  # 데이터셋 변환
│   └── verify_merge.py      # Merge 검증
├── configs/
│   ├── ds_zero1.json         # DeepSpeed ZeRO-1 기본
│   └── ds_zero1_sift.json    # DeepSpeed ZeRO-1 + optimizer/scheduler (sparse method용)
├── run_all.sh         # 전체 실험 파이프라인 (MT-Bench → MMLU → CSR)
├── run_mtbench.sh     # MT-Bench only
└── smoke_test.sh      # 각 method smoke test (20 steps)
```

## Key Design: DeepSpeed 8-GPU Compatible Sparse Training

모든 sparse method가 DeepSpeed ZeRO-1 + 8 GPU에서 동작하도록 통일된 패턴 적용:

```python
class SparseLinear(nn.Module):
    """Drop-in replacement for nn.Linear with sparse trainable delta."""
    def __init__(self, orig_linear, flat_idx):
        super().__init__()
        # Frozen weight → buffer (DeepSpeed가 optimizer에서 제외)
        self.register_buffer("weight", orig_linear.weight.data)
        self.register_buffer("flat_idx", flat_idx)
        # Trainable sparse delta → Parameter (DeepSpeed가 optimize)
        self.sparse_delta = nn.Parameter(torch.zeros(len(flat_idx), ...))
    
    def forward(self, x):
        delta_flat = torch.zeros(self.weight.numel(), ...)
        delta_flat.scatter_(0, self.flat_idx, self.sparse_delta)
        return F.linear(x, self.weight + delta_flat.view(self.weight.shape), self.bias)
```

**핵심**: `weight`를 `register_buffer`로 등록해서 DeepSpeed가 gradient buffer를 할당하지 않음 → OOM 방지.

## Setup

```bash
# 1. 가상환경 (반드시 /home1/irteam에 생성 — OOM killed 방지)
python -m venv /home1/irteam/rapa/.rapa
source /home1/irteam/rapa/.rapa/bin/activate

# 2. 의존성
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate datasets peft trl deepspeed sentencepiece openai vllm evaluate lm-eval

# 3. LMFlow clone
cd /home1/irteam/rapa
git clone https://github.com/OptimalScale/LMFlow.git

# 4. sparse_ft 코드를 LMFlow에 연결
cp -r sparse_ft/pipeline/* LMFlow/src/lmflow/pipeline/rapa/
cp -r sparse_ft/configs/* LMFlow/configs/rapa/

# 5. SpiEL peft fork의 linear_sd C extension 빌드
cd methods/spiel/peft_sft/linear-sd
python setup.py install
```

## Running Experiments

### Smoke Test (각 method 동작 확인, 20 steps)
```bash
cd /home1/irteam/rapa/LMFlow

# 개별 method
bash scripts/rapa/smoke_test.sh sift
bash scripts/rapa/smoke_test.sh spiel
bash scripts/rapa/smoke_test.sh smt
bash scripts/rapa/smoke_test.sh s2ft
bash scripts/rapa/smoke_test.sh ltsft
```

### Full Training + Evaluation
```bash
# 전체 실험 (MT-Bench → MMLU → CSR), nohup으로 서버 끊겨도 유지
cd /home1/irteam/rapa/LMFlow
nohup bash scripts/rapa/run_all.sh > /home1/irteam/rapa/results/run_all.log 2>&1 &

# 진행상황 확인
tail -f /home1/irteam/rapa/results/run_all.log
```

### Individual Training
```bash
# 단일 method 학습 (deepspeed 8 GPU)
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=29600 \
    examples/rapa/train_method.py \
    --method sift \
    --model_name_or_path mistralai/Mistral-7B-v0.3 \
    --dataset_path /home1/irteam/rapa/data/oasst1_lmflow.json \
    --output_dir /home1/irteam/rapa/checkpoints/mtbench_sift \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --max_seq_length 512 \
    --target_params 170000000 \
    --bf16 \
    --hf_token YOUR_HF_TOKEN
```

## Hyperparameters

### MT-Bench Task
| Param | Value |
|-------|-------|
| Model | `mistralai/Mistral-7B-v0.3` |
| Dataset | `timdettmers/openassistant-guanaco` (oasst1) |
| Batch size/GPU | 1 |
| Grad accum | 1 |
| LR scheduler | linear |
| Learning rate | 5e-5 |
| Epoch | 1 |
| Max seq length | 512 |
| # Trainable params | 170M |
| Judge | GPT-4o-mini |

### MMLU (5-shot) / CSR (0-shot)
| Param | Value |
|-------|-------|
| Model | `meta-llama/Llama-2-7b-hf` |
| Dataset | `/home1/irteam/datasets/mmlu/mmlu.json`, `/home1/irteam/datasets/merge/merge.json` |
| LR scheduler | cosine |
| Learning rate | 5e-5 |
| Epoch | 1 |
| Max seq length | 512 |
| # Trainable params | 170M |

## SIFT Index Caching

SIFT sparse index 생성은 7B 모델에서 시간이 걸립니다. 최초 1회 생성 후 `.pt` 파일로 캐시:
- 캐시 위치: `/home1/irteam/rapa/checkpoints/.sift_idx_cache/`
- 캐시 키: `hash(model_name + sparse_rate + modules + seed + dataset_name)`
- 같은 모델 + 같은 설정이면 데이터셋이 달라도 `torch.load`로 즉시 로드

## Important Notes

1. **OOM 방지**: 모든 데이터(모델, 캐시, 체크포인트)를 `/home1/irteam/rapa/`에 저장
2. **8 GPU 필수**: DeepSpeed ZeRO-1으로 8 GPU 학습, vLLM tensor_parallel로 8 GPU 평가
3. **DeepSpeed config**: `ds_zero1_sift.json`에 optimizer + scheduler 명시 필요 (frozen params와의 호환성)
4. **SpiEL fork**: `AlanAnsell/peft` fork 필요, `linear_sd` C extension 빌드 필수
