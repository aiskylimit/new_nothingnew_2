# commands.sh - cac lenh chay tren server offline, theo tung giai doan.
# Chay:  cd SegmentSelectiveSFT && bash commands.sh
# Moi duong dan tuong doi ben duoi (data/, SelectiveSFT/, Eval/) tinh tu repo
# root; dong cd duoi day bao dam dieu do ke ca khi goi tu thu muc khac.
# Moi giai doan dung MOT env rieng (source dung dong roi chay tiep), khong tron:
#   ssft_eval   : split / ig / segments + eval.sh (vllm 0.10.2 + torch 2.8.0) <- ../ssft_eval.txt
#                 (torch 2.7.1 wheel PyPI khong co kernel sm_100/B200 -> phai len 2.8.0 cu128)
#   ssft_train  : train.sh + merge_lora.py (torch 2.9 + unsloth + peft)      <- ../ssft_train.txt

cd "$(dirname "${BASH_SOURCE[0]}")"
# Dung ngay khi mot buoc loi (setup check thieu goi, train OOM, khong co checkpoint...)
# thay vi chay tiep sang eval voi checkpoint cu.
set -eo pipefail

PROJECT=aiskylimit_new_nothingnew_2          # = @PROJECT@ trong downloads.txt
MODEL_DIR=/mnt/local/_models/$PROJECT/Qwen2.5-7B-Instruct
# Model tinh IG: paper (App C.3) luon dung R1-Distill-Qwen-7B (model sinh CoT),
# ke ca khi train Qwen2.5-7B-Instruct. MODEL_DIR o tren chi dung cho stage train/eval.
ATTR_MODEL_DIR=/mnt/local/_models/$PROJECT/DeepSeek-R1-Distill-Qwen-7B
DATA_DIR=/mnt/local/_data/$PROJECT/s1k
# Benchmark eval tai ve dang HF dataset (downloads.txt):
#   $EVAL_DATA_ROOT/aime24  aime25  MATH-500  aimo-validation-amc
# prepare_eval_data.py chuyen thanh data/<task>/test.jsonl (question + answer).
EVAL_DATA_ROOT=/mnt/local/_data/$PROJECT

# =============================================================================
# [1] ATTRIBUTION (da chay xong, giu lai de tham khao) - env ssft_eval
# =============================================================================
# source /mnt/local/uvenvs/ssft_eval/bin/activate
# ls "$DATA_DIR" "$ATTR_MODEL_DIR"
# mkdir -p data/s1k && ln -sf "$DATA_DIR/train.jsonl" data/s1k/train.jsonl
# bash run_pipeline.sh --offline --attr-model "$ATTR_MODEL_DIR" --gpu-attr 0 --ig-no-grad-checkpoint --ig-batch-size 2 --stages split,ig,segments
# bash make_bundle.sh          # -> bundle/ de tai ve may khac

# =============================================================================
# [2] SELECTIVE SFT - env ssft_train  (lan 4: full finetuning, effective batch 16 = 2 x 8)
#     Lan 3 (batch 8) da xong, checkpoint o
#     SelectiveSFT/checkpoints/Qwen2.5-7B-Instruct_epoch3_lr5e-5_len32768/ (eval: sel_ft_ep3_8k).
#     Lan nay them --ckpt-suffix _bs16 -> thu muc ..._len32768_bs16 rieng, khong de
#     len ban batch 8 (ten thu muc khong chua batch; HF save_total_limit=3 se xoa
#     dan checkpoint cu neu dung chung thu muc).
#     Cac ban LoRA r=16 / r=64 nam o ..._lora / ..._lora_r64, eval bang --model.
# =============================================================================
source /mnt/local/uvenvs/ssft_train/bin/activate
bash setup.sh check --for train          # phai thay torch 2.9 / unsloth / peft / torchao<0.18

# Neu solutions_selected.jsonl la ban ghep tu bundle (xem bundle/GHEP_LAI.txt):
#   cat solutions_selected.jsonl.part* > solutions_selected.jsonl
mkdir -p data/s1k
[[ -f data/s1k/solutions_selected.jsonl ]] || ln -sf "$DATA_DIR/solutions_selected.jsonl" data/s1k/solutions_selected.jsonl
wc -l data/s1k/solutions_selected.jsonl

# Cau hinh train (ghi tuong minh):
#   full finetuning toan bo 7B (unsloth full_finetuning=True), khong adapter
#   effective batch = 2 per-device x 8 accum x 1 GPU = 16 mau/step
#   -> ~1000 mau / 16 = ~63 step/epoch, 3 epoch ~ 189 step (lan batch 8: 375 step)
#   per-device 2: moi micro-step 2 mau pad theo mau dai hon; khong dung
#   --group-by-length de sampling ngau nhien hoan toan nhu lan batch 8
#   AdamW betas (0.9, 0.999) eps 1e-8 (mac dinh) weight_decay 0.0
#   cosine + warmup, warmup_ratio 0.1 (HF dung LambdaLR)
#   max_seq_length 32768; segment = paragraph: moi doan "\n\n" = 1 reasoning step
#   gradient checkpointing: bat (mac dinh) - bat buoc o 32k tren 7B
# VRAM (GPU 0, 180 GB): 7.6B x 16 byte (weight bf16 + grad + 2 state AdamW fp32
# + master) ~ 120 GB co dinh. Phan theo batch TANG GAP DOI so voi per-device 1:
# activation o 32k voi grad checkpointing ~ 20 GB; logits 2 x 32768 x 152k vocab
# fp32 ~ 40 GB (+ grad) -> tong ~ 180-200 GB, NHIEU KHA NANG OOM. Neu OOM:
#   1) --optim adamw_8bit (van AdamW cung betas/eps/wd, state 8-bit, giam ~45 GB)
#   2) quay ve --batch-size 1 --grad-accum 16 (cung 16 mau/step, ket qua tuong duong)
# KHONG giam --max-seq-length vi se cat mat response cua mau dai.
TRAIN_ARGS=(
  --offline --model "$MODEL_DIR" --gpu 0
  --full-finetune
  --epochs 3 --lr 5e-5 --max-seq-length 32768
  --batch-size 2 --grad-accum 8 --ckpt-suffix _bs16
  --optim adamw_torch --weight-decay 0.0 --lr-scheduler cosine --warmup-ratio 0.1
  --segment-mode paragraph
)
bash train.sh "${TRAIN_ARGS[@]}" --dry-run     # in lenh truoc, chua chay
bash train.sh "${TRAIN_ARGS[@]}"               # log: logs/train_bs16.log
# Baseline de so sanh: SFT tren toan bo CoT (khong mask)
# bash train.sh "${TRAIN_ARGS[@]}" --full-sft

# Full finetuning luu thang weight day du -> KHONG can merge_lora.py.
CKPT_DIR=SelectiveSFT/checkpoints/$(basename "$MODEL_DIR")_epoch3_lr5e-5_len32768_bs16
CKPT=$(ls -1d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
echo "checkpoint moi nhat: $CKPT_DIR/checkpoint-$CKPT"
ls "$CKPT_DIR/checkpoint-$CKPT"                  # phai co config.json + model*.safetensors
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
# Selective SFT full-finetune batch 16: --full-finetune + --ckpt-suffix _bs16 de eval.sh tim
# ..._len32768_bs16 (khong hau to _lora_r<R>); checkpoint la weight day du nen khong can merge.
# Tag rieng (sel_ft_ep3_bs16_32k) de khong tron voi outputs_sel_ft_ep3_8k cua ban batch 8:
# summary.json va pass_at_k.py gom MOI file duoi outputs_<tag>.
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --full-finetune --ckpt-suffix _bs16 --tag sel_ft_ep3_bs16_32k --dry-run
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --full-finetune --ckpt-suffix _bs16 --tag sel_ft_ep3_bs16_32k
# Ban full-finetune batch 8 (lan 3, thu muc khong hau to):
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --full-finetune --tag sel_ft_ep3_32k   # (ban 8192 cu: outputs_sel_ft_ep3_8k)
# Ban LoRA r=64 (da merge, --lora-r mac dinh 64):
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --tag sel_r64_ep3
# Ban r=16 cu (thu muc ten cu, khong co _r16):
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --model SelectiveSFT/checkpoints/Qwen2.5-7B-Instruct_epoch3_lr5e-5_len32768_lora/checkpoint-90-merged --tag sel_ep3
# Base chua finetune
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --base --tag base
# Baseline full-CoT SFT
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --full-sft --full-finetune --tag fullsft_ft_ep3
# Nhieu GPU: them --gpu 0,1,2,3 --data-parallel (1 task / GPU)

# "acc" trong summary.json chi tinh MAU DAU TIEN moi cau (evaluate.py: mean_score[0]).
# Pass@1 / Pass@3 unbiased tren du 3 mau + AVG macro qua 4 bo:
(cd Eval && python pass_at_k.py outputs_sel_ft_ep3_bs16_32k --k 1 3)
# (cd Eval && python pass_at_k.py outputs_sel_ft_ep3_8k --k 1 3)    # ban batch 8
# (cd Eval && python pass_at_k.py outputs_sel_r64_ep3 --k 1 3)  # ban LoRA r=64
# (cd Eval && python pass_at_k.py outputs_sel_ep3 --k 1 3)     # ban r=16
# (cd Eval && python pass_at_k.py outputs_base --k 1 3)
