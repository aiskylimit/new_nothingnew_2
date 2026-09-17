# commands.sh - chay day du full-CoT SFT + eval tren server offline.
# Chay:  cd SegmentSelectiveSFT && bash commands.sh
# Moi duong dan tuong doi ben duoi (data/, SelectiveSFT/, Eval/) tinh tu repo
# root; dong cd duoi day bao dam dieu do ke ca khi goi tu thu muc khac.
# Moi giai doan dung MOT env rieng (source dung dong roi chay tiep), khong tron:
#   ssft_train  : prepare_s1k.py + train.sh + merge_lora.py (torch 2.9 + unsloth + peft) <- ../ssft_train.txt
#   ssft_eval   : eval.sh (vllm 0.10.2 + torch 2.8.0)                                    <- ../ssft_eval.txt
#                 (torch 2.7.1 wheel PyPI khong co kernel sm_100/B200 -> phai len 2.8.0 cu128)

cd "$(dirname "${BASH_SOURCE[0]}")"
# Dung ngay khi mot buoc loi (setup check thieu goi, train OOM, khong co checkpoint...)
# thay vi chay tiep sang eval voi checkpoint cu.
set -eo pipefail

PROJECT=aiskylimit_new_nothingnew_2          # = @PROJECT@ trong downloads.txt
MODEL_DIR=/mnt/local/_models/$PROJECT/Qwen2.5-7B-Instruct
# Snapshot baesad/s1K-1.1-deepseek-cot (downloads.txt): da co san $DATA_DIR/train.jsonl dung format
# pipeline (question / solution=deepseek_thinking_trajectory / answer) -> chi can symlink.
DATA_DIR=/mnt/local/_data/$PROJECT/s1k
# Benchmark eval tai ve dang HF dataset (downloads.txt):
#   $EVAL_DATA_ROOT/aime24  aime25  MATH-500  aimo-validation-amc
# prepare_eval_data.py chuyen thanh data/<task>/test.jsonl (question + answer).
EVAL_DATA_ROOT=/mnt/local/_data/$PROJECT

# Ten run - dung chung cho tag eval va thu muc outputs_<tag>.
TAG=fullsft_r16_ep3

# =============================================================================
# [1] PREP - env ssft_train: $DATA_DIR/train.jsonl -> data/s1k/train.jsonl
# =============================================================================
source /mnt/local/uvenvs/ssft_train/bin/activate
bash setup.sh check --for train          # phai thay torch 2.9 / unsloth / peft / torchao<0.18
ls "$DATA_DIR" "$MODEL_DIR"

mkdir -p data/s1k
if [[ ! -s data/s1k/train.jsonl ]]; then
  if [[ -s "$DATA_DIR/train.jsonl" ]]; then
    ln -sf "$DATA_DIR/train.jsonl" data/s1k/train.jsonl
  else
    # Snapshot goc simplescaling/s1K-1.1 (parquet) thi phai doi format:
    # solution <- deepseek_thinking_trajectory (trace R1), answer <- \boxed{} cuoi trace.
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python prepare_s1k.py \
      --dataset "$DATA_DIR" --output_data_file data/s1k/train.jsonl
  fi
fi
wc -l data/s1k/train.jsonl               # ~1000 dong
head -c 300 data/s1k/train.jsonl; echo   # phai thay "question" / "solution" / "answer"

# =============================================================================
# [2] TRAIN - full-CoT SFT (supervise TOAN BO trace, khong mask), LoRA r=16 - env ssft_train
# =============================================================================
# Cau hinh (ghi tuong minh):
#   LoRA r 16 / alpha 16 / dropout 0.05, target q,k,v,o,gate,up,down (mac dinh train.sh)
#   effective batch = 1 per-device x 32 accum x 1 GPU = 32 mau/step
#   -> ~1000 mau / 32 = ~32 step/epoch, 3 epoch ~ 96 step
#   AdamW betas (0.9, 0.999) eps 1e-8 (mac dinh) weight_decay 0.0
#   cosine + warmup, warmup_ratio 0.1 (HF dung LambdaLR)
#   max_seq_length 32768 (= max_position_embeddings cua Qwen2.5-7B); gradient checkpointing bat
# Khong --mask nen khong can selected_spans_ids -> train thang tren train.jsonl, khong can chay IG.
# Thu muc checkpoint: SelectiveSFT/checkpoints/Qwen2.5-7B-Instruct_epoch3_lr5e-5_len32768_fullsft_lora_r16
# (train.sh tu ghep hau to _fullsft + _lora_r16; eval.sh --full-sft --lora-r 16 tim dung thu muc nay).
# VRAM: weight bf16 ~15 GB + adapter r16 nho, khong co optimizer state cua 7B nhu full finetune;
# nang nhat la logits 32768 x 152k vocab fp32 ~ 20 GB (+ grad) -> ~60-80 GB, du cho 1 GPU 80 GB.
# KHONG giam --max-seq-length vi se cat mat response cua mau dai (train_mask.py bo mau do, bao so luong).
FULLSFT_ARGS=(
  --offline --model "$MODEL_DIR" --gpu 0
  --full-sft --data data/s1k/train.jsonl
  --lora-r 16 --lora-alpha 16 --lora-dropout 0.05
  --epochs 3 --lr 5e-5 --max-seq-length 32768
  --batch-size 1 --grad-accum 32
  --optim adamw_torch --weight-decay 0.0 --lr-scheduler cosine --warmup-ratio 0.1
)
bash train.sh "${FULLSFT_ARGS[@]}" --dry-run   # in lenh truoc, chua chay
bash train.sh "${FULLSFT_ARGS[@]}"             # log: logs/train_fullsft_lora_r16.log

# Checkpoint la adapter LoRA -> merge vao weight goc truoc khi eval (peft chi co o env train; CPU du).
CKPT_DIR=SelectiveSFT/checkpoints/$(basename "$MODEL_DIR")_epoch3_lr5e-5_len32768_fullsft_lora_r16
CKPT=$(ls -1d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
[[ -n "$CKPT" ]] || { echo "Khong thay checkpoint trong $CKPT_DIR"; exit 1; }
echo "checkpoint moi nhat: $CKPT_DIR/checkpoint-$CKPT"
if [[ ! -f "$CKPT_DIR/checkpoint-$CKPT-merged/config.json" ]]; then
  (cd SelectiveSFT && HF_HUB_OFFLINE=1 python merge_lora.py \
      --adapter "../$CKPT_DIR/checkpoint-$CKPT" --base_model "$MODEL_DIR")
fi
ls "$CKPT_DIR/checkpoint-$CKPT-merged"          # phai co config.json + model*.safetensors
deactivate

# =============================================================================
# [3] EVAL - env ssft_eval (KHONG dung ssft_train)
# =============================================================================
# Thiet lap: t=0.6, top_p=0.9, repetition_penalty=1.05, max_tokens=32768 (= max_seq_length luc train;
# truoc 4096 roi 8192: CoT dai van bi cat mat \boxed -> tinh sai),
# k=3 mau/cau cho MOI benchmark (aime24 aime25 amc12 math500).
# amc12 = AI-MO/aimo-validation-amc (83 cau AMC12 2022-2023).
# Pass@1 = trung binh acc tren 3 mau; Pass@3 = 1 neu bat ky mau nao dung;
# AVG = trung binh cong don gian qua 4 bo (pass_at_k.py tinh, xem cuoi file).
source /mnt/local/uvenvs/ssft_eval/bin/activate
bash setup.sh check --for eval           # phai thay vllm 0.10.2 / torch 2.8.0 / latex2sympy
python prepare_eval_data.py --data-root "$EVAL_DATA_ROOT"   # bo qua task da co test.jsonl
wc -l data/aime24/test.jsonl data/aime25/test.jsonl data/amc12/test.jsonl data/math500/test.jsonl

EVAL_ARGS=(
  --offline --gpu 0
  --tasks "aime24 aime25 amc12 math500" --n-sampling 3
  --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 --max-tokens 32768
)
# --full-sft --lora-r 16: eval.sh tim ..._fullsft_lora_r16/checkpoint-<moi nhat>, thay adapter thi tu
# doi sang ban -merged. eval.sh resumable: task da co *_metrics.json thi bo qua (--overwrite de cham lai).
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --full-sft --lora-r 16 --tag "$TAG" --dry-run
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --full-sft --lora-r 16 --tag "$TAG"
# Base chua finetune, de so sanh:
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --base --tag base
# Nhieu GPU: them --gpu 0,1,2,3 --data-parallel (1 task / GPU)

# "acc" trong summary.json chi tinh MAU DAU TIEN moi cau (evaluate.py: mean_score[0]).
# Pass@1 / Pass@3 unbiased tren du 3 mau + AVG macro qua 4 bo:
(cd Eval && python pass_at_k.py "outputs_$TAG" --k 1 3)
# (cd Eval && python pass_at_k.py outputs_base --k 1 3)
deactivate

# =============================================================================
# [Tham khao] Selective SFT (paper): can attribution truoc, env ssft_eval cho IG
# =============================================================================
# ATTR_MODEL_DIR=/mnt/local/_models/$PROJECT/DeepSeek-R1-Distill-Qwen-7B   # paper App C.3: model sinh CoT
# source /mnt/local/uvenvs/ssft_eval/bin/activate
# bash run_pipeline.sh --offline --attr-model "$ATTR_MODEL_DIR" --gpu-attr 0 --ig-no-grad-checkpoint --ig-batch-size 2 --stages split,ig,segments
# source /mnt/local/uvenvs/ssft_train/bin/activate
# bash train.sh "${FULLSFT_ARGS[@]/--full-sft/--selective}" --data data/s1k/solutions_selected.jsonl   # cung r=16 / batch 32
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --lora-r 16 --tag sel_r16_ep3
