# commands.sh - cac lenh chay tren server offline, theo tung giai doan.
# Moi giai doan dung MOT env rieng (source dung dong roi chay tiep), khong tron:
#   ssft_eval   : split / ig / segments + eval.sh (vllm 0.10 + torch 2.7.1)  <- ../ssft_eval.txt
#   ssft_train  : train.sh + merge_lora.py (torch 2.9 + unsloth + peft)      <- ../ssft_train.txt

MODEL_DIR=/mnt/local/_models/aiskylimit_new_nothingnew_2/Qwen2.5-7B-Instruct
# Model tinh IG: paper (App C.3) luon dung R1-Distill-Qwen-7B (model sinh CoT),
# ke ca khi train Qwen2.5-7B-Instruct. MODEL_DIR o tren chi dung cho stage train/eval.
ATTR_MODEL_DIR=/mnt/local/_models/aiskylimit_new_nothingnew_2/DeepSeek-R1-Distill-Qwen-7B
DATA_DIR=/mnt/local/_data/aiskylimit_new_nothingnew_2/s1k
# Test set cua eval (<task>/test.jsonl, truong question + answer). Repo khong
# ship nua; tro vao ban da tai ve, hoac copy thang vao data/<task>/test.jsonl.
EVAL_DATA_DIR=/mnt/local/_data/aiskylimit_new_nothingnew_2/eval_bench

# =============================================================================
# [1] ATTRIBUTION (da chay xong, giu lai de tham khao) - env ssft_eval
# =============================================================================
# source /mnt/local/uvenvs/ssft_eval/bin/activate
# ls "$DATA_DIR" "$ATTR_MODEL_DIR"
# mkdir -p data/s1k && ln -sf "$DATA_DIR/train.jsonl" data/s1k/train.jsonl
# bash run_pipeline.sh --offline --attr-model "$ATTR_MODEL_DIR" --gpu-attr 0 --ig-no-grad-checkpoint --ig-batch-size 2 --stages split,ig,segments
# bash make_bundle.sh          # -> bundle/ de tai ve may khac

# =============================================================================
# [2] SELECTIVE SFT - env ssft_train
# =============================================================================
source /mnt/local/uvenvs/ssft_train/bin/activate
bash setup.sh check --for train          # phai thay torch 2.9 / unsloth / peft / torchao<0.18

# Neu solutions_selected.jsonl la ban ghep tu bundle (xem bundle/GHEP_LAI.txt):
#   cat solutions_selected.jsonl.part* > solutions_selected.jsonl
mkdir -p data/s1k
[[ -f data/s1k/solutions_selected.jsonl ]] || ln -sf "$DATA_DIR/solutions_selected.jsonl" data/s1k/solutions_selected.jsonl
wc -l data/s1k/solutions_selected.jsonl

# Cau hinh train (ghi tuong minh, trung voi mac dinh cua train.sh):
#   LoRA r=16 alpha=16 dropout=0.05 tren q/k/v/o/gate/up/down_proj
#   effective batch = 1 x 32 accum x 1 GPU = 32 mau/step
#   AdamW betas (0.9, 0.999) eps 1e-8 (mac dinh) weight_decay 0.0
#   cosine + warmup, warmup_ratio 0.1 (HF dung LambdaLR)
#   max_seq_length 32768; segment = paragraph: moi doan "\n\n" = 1 reasoning step
TRAIN_ARGS=(
  --offline --model "$MODEL_DIR" --gpu 0
  --lora --lora-r 16 --lora-alpha 16 --lora-dropout 0.05
  --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
  --epochs 3 --lr 5e-5 --max-seq-length 32768
  --batch-size 1 --grad-accum 32
  --optim adamw_torch --weight-decay 0.0 --lr-scheduler cosine --warmup-ratio 0.1
  --segment-mode paragraph
)
bash train.sh "${TRAIN_ARGS[@]}" --dry-run     # in lenh truoc, chua chay
bash train.sh "${TRAIN_ARGS[@]}"               # log: logs/train_lora.log
# Baseline de so sanh: SFT tren toan bo CoT (khong mask)
# bash train.sh "${TRAIN_ARGS[@]}" --full-sft

# Merge adapter LoRA -> checkpoint-<step>-merged (PHAI o env train vi can peft; CPU la du)
CKPT_DIR=SelectiveSFT/checkpoints/$(basename "$MODEL_DIR")_epoch3_lr5e-5_len32768_lora
CKPT=$(ls -1d "$CKPT_DIR"/checkpoint-* 2>/dev/null | grep -v merged | sed 's#.*/checkpoint-##' | sort -n | tail -1)
echo "checkpoint moi nhat: $CKPT_DIR/checkpoint-$CKPT"
(cd SelectiveSFT && python merge_lora.py --adapter "../$CKPT_DIR/checkpoint-$CKPT" --base_model "$MODEL_DIR")
deactivate

# =============================================================================
# [3] EVAL - env ssft_eval (KHONG dung ssft_train)
# =============================================================================
# Thiet lap: t=0.6, top_p=0.9, repetition_penalty=1.05, max_tokens=4096,
# k=3 mau/cau cho MOI benchmark (aime24 aime25 amc23 math500).
# AMC23 = AMC12 nam 2023 (40 cau); repo dat ten task la amc23.
# Pass@1 = trung binh acc tren 3 mau; Pass@3 = 1 neu bat ky mau nao dung;
# AVG = trung binh cong don gian qua 4 bo (pass_at_k.py tinh, xem cuoi file).
source /mnt/local/uvenvs/ssft_eval/bin/activate
bash setup.sh check --for eval           # phai thay vllm 0.10 / torch 2.7.1 / latex2sympy
for t in aime24 aime25 amc23 math500; do
  [[ -f data/$t/test.jsonl ]] || { mkdir -p data/$t && ln -sf "$EVAL_DATA_DIR/$t/test.jsonl" data/$t/test.jsonl; }
done
ls -l data/aime24 data/aime25 data/amc23 data/math500

EVAL_ARGS=(
  --offline --gpu 0
  --tasks "aime24 aime25 amc23 math500" --n-sampling 3
  --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 --max-tokens 4096
)
# Selective SFT (tu tim checkpoint-<step>-merged trong CKPT_DIR)
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --tag sel_ep3 --dry-run
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --tag sel_ep3
# Base chua finetune
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --base --tag base
# Baseline full-CoT SFT
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --full-sft --tag fullsft_ep3
# Nhieu GPU: them --gpu 0,1,2,3 --data-parallel (1 task / GPU)

# "acc" trong summary.json chi tinh MAU DAU TIEN moi cau (evaluate.py: mean_score[0]).
# Pass@1 / Pass@3 unbiased tren du 3 mau + AVG macro qua 4 bo:
(cd Eval && python pass_at_k.py outputs_sel_ep3 --k 1 3)
# (cd Eval && python pass_at_k.py outputs_base --k 1 3)
