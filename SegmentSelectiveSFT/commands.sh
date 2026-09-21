# commands.sh - chay day du full-CoT SFT + eval cho Qwen3-8B tren server offline.
# Chay:  cd SegmentSelectiveSFT && bash commands.sh
# Moi duong dan tuong doi ben duoi (data/, SelectiveSFT/, Eval/) tinh tu repo
# root; dong cd duoi day bao dam dieu do ke ca khi goi tu thu muc khac.
# Moi giai doan dung MOT env rieng (source dung dong roi chay tiep), khong tron:
#   ssft_train  : prepare_s1k.py + train.sh + merge_lora.py (torch 2.9 + unsloth + peft) <- ../ssft_train.txt
#   ssft_eval   : eval.sh (vllm 0.10.2 + torch 2.8.0)                                    <- ../ssft_eval.txt
#                 (torch 2.7.1 wheel PyPI khong co kernel sm_100/B200 -> phai len 2.8.0 cu128)
# Ca hai env deu da ho tro kien truc Qwen3 (transformers 4.57 / unsloth / vllm >= 0.8.5).

cd "$(dirname "${BASH_SOURCE[0]}")"
# Dung ngay khi mot buoc loi (setup check thieu goi, train OOM, khong co checkpoint...)
# thay vi chay tiep sang eval voi checkpoint cu.
set -eo pipefail

PROJECT=aiskylimit_new_nothingnew_2          # = @PROJECT@ trong downloads.txt
# Qwen/Qwen3-8B (downloads.txt): ban post-trained co thinking mode, 36 layer, hidden 4096,
# max_position_embeddings 40960 (> 32768 nen giu nguyen max_seq_length / max_tokens ben duoi).
MODEL_DIR=/mnt/local/_models/$PROJECT/Qwen3-8B
# Snapshot baesad/s1K-1.1-deepseek-cot (downloads.txt): da co san $DATA_DIR/train.jsonl dung format
# pipeline (question / solution=deepseek_thinking_trajectory / answer) -> chi can symlink.
DATA_DIR=/mnt/local/_data/$PROJECT/s1k
# Benchmark eval tai ve dang HF dataset (downloads.txt):
#   $EVAL_DATA_ROOT/aime24  aime25  MATH-500  aimo-validation-amc
# prepare_eval_data.py chuyen thanh data/<task>/test.jsonl (question + answer).
EVAL_DATA_ROOT=/mnt/local/_data/$PROJECT

# GPU dung cho ca train lan eval (train.sh / eval.sh export CUDA_VISIBLE_DEVICES=$GPU).
# Mot so: "1" = chi GPU 1. Eval nhieu GPU: GPU=0,1,2,3 + them --data-parallel vao EVAL_ARGS (1 task/GPU).
# Train luon 1 GPU (1 tien trinh); effective batch = 1 x 32 accum = 32 khong doi.
GPU=${GPU:-1}

# Ten run - dung chung cho tag eval va thu muc outputs_<tag>.
# Doi ten so voi run Qwen2.5 truoc (fullsft_r16_ep3) de hai bo output khong de len nhau.
TAG=qwen3_8b_fullsft_r16_ep3
BASE_TAG=qwen3_8b_base

# =============================================================================
# [1] PREP - env ssft_train: $DATA_DIR/train.jsonl -> data/s1k/train.jsonl
# =============================================================================
source /mnt/local/uvenvs/ssft_train/bin/activate
bash setup.sh check --for train          # phai thay torch 2.9 / unsloth / peft / torchao<0.18
ls "$DATA_DIR" "$MODEL_DIR"
# Qwen3-8B phai co tokenizer.json (fast tokenizer) - train_mask.py can offset_mapping de mask segment.
[[ -f "$MODEL_DIR/tokenizer.json" ]] || { echo "Thieu $MODEL_DIR/tokenizer.json"; exit 1; }

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
#   max_seq_length 32768 (< 40960 cua Qwen3-8B, khong can RoPE scaling); gradient checkpointing bat
# Chat template Qwen3: apply_chat_template(add_generation_prompt=True) mac dinh (thinking mode) ket thuc
# bang "<|im_start|>assistant\n" - khop response_template cua train_mask.py voi --think-prefix none
# (train_mask.py tu probe template va dung ngay neu lech). KHONG dung --think-prefix plain/special:
# Qwen3 da co san token <think>/</think> nhung trace s1K la text tran, khong co the <think>,
# va prompt luc eval (math_eval.py --apply_chat_template, khong --enable_think) cung khong chen <think>.
# Khong --mask nen khong can selected_spans_ids -> train thang tren train.jsonl, khong can chay IG.
# Thu muc checkpoint: SelectiveSFT/checkpoints/Qwen3-8B_epoch3_lr5e-5_len32768_fullsft_lora_r16
# (train.sh tu ghep hau to _fullsft + _lora_r16; eval.sh --full-sft --lora-r 16 tim dung thu muc nay).
# VRAM: weight bf16 ~16.4 GB (8.2B) + adapter r16 nho, khong co optimizer state cua 8B nhu full finetune;
# nang nhat la logits 32768 x 152k vocab fp32 ~ 20 GB (+ grad) -> ~65-80 GB, du cho 1 GPU 80 GB
# (36 layer, hidden 4096 -> activation nhinh hon Qwen2.5-7B mot chut; neu OOM: --load-4bit hoac --lora-r 8,
# KHONG giam --max-seq-length vi se cat mat response cua mau dai - train_mask.py bo mau do, bao so luong).
FULLSFT_ARGS=(
  --offline --model "$MODEL_DIR" --gpu "$GPU"
  --full-sft --data data/s1k/train.jsonl
  --lora-r 16 --lora-alpha 16 --lora-dropout 0.05
  --epochs 3 --lr 5e-5 --max-seq-length 32768
  --batch-size 1 --grad-accum 32
  --optim adamw_torch --weight-decay 0.0 --lr-scheduler cosine --warmup-ratio 0.1
  --think-prefix none
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
# vLLM: max_model_len tu lay 40960 tu config Qwen3-8B -> prompt + 32768 token sinh ra van vua;
# stop_token_ids <|im_end|>=151645 / <|endoftext|>=151643 giong Qwen2.5 (math_eval.py bat theo "qwen" trong ten).
# Model base Qwen3-8B (thinking mode) tu sinh <think>...</think> roi moi \boxed; model da SFT hoc trace tran
# nen thuong khong sinh the <think> - ca hai deu ket thuc bang \boxed{} nen grader cham nhu nhau.
source /mnt/local/uvenvs/ssft_eval/bin/activate
bash setup.sh check --for eval           # phai thay vllm 0.10.2 / torch 2.8.0 / latex2sympy
python prepare_eval_data.py --data-root "$EVAL_DATA_ROOT"   # bo qua task da co test.jsonl
wc -l data/aime24/test.jsonl data/aime25/test.jsonl data/amc12/test.jsonl data/math500/test.jsonl

EVAL_ARGS=(
  --offline --gpu "$GPU"
  --tasks "aime24 aime25 amc12 math500" --n-sampling 3
  --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 --max-tokens 32768
)
# --full-sft --lora-r 16: eval.sh tim ..._fullsft_lora_r16/checkpoint-<moi nhat>, thay adapter thi tu
# doi sang ban -merged. eval.sh resumable: task da co *_metrics.json thi bo qua (--overwrite de cham lai).
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --full-sft --lora-r 16 --tag "$TAG" --dry-run
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --full-sft --lora-r 16 --tag "$TAG"
# Base Qwen3-8B chua finetune, de so sanh (ton them ~1 lan eval nua; bo comment neu muon):
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --base --tag "$BASE_TAG"
# Nhieu GPU: dat GPU=0,1,2,3 o dau file va them --data-parallel vao EVAL_ARGS (1 task / GPU)

# "acc" trong summary.json chi tinh MAU DAU TIEN moi cau (evaluate.py: mean_score[0]).
# Pass@1 / Pass@3 unbiased tren du 3 mau + AVG macro qua 4 bo (ghi <root>/pass_at_k.json):
(cd Eval && python pass_at_k.py "outputs_$TAG" --k 1 3)
[[ -d "Eval/outputs_$BASE_TAG" ]] && (cd Eval && python pass_at_k.py "outputs_$BASE_TAG" --k 1 3)

# =============================================================================
# [4] KET QUA - in bang tong hop ra man hinh (doc lai summary.json + pass_at_k.json)
# =============================================================================
# Chay rieng buoc nay (khong train/eval lai) khi chi muon xem lai ket qua:
#   TAG=qwen3_8b_fullsft_r16_ep3 python - <<'PY' ... (copy khoi ben duoi)
python - "$MODEL_DIR" "$TAG" "$BASE_TAG" <<'PY'
import json, os, sys
model_dir, tag, base_tag = sys.argv[1:4]
runs = [(tag, "full-CoT SFT LoRA r16, 3 epoch"), (base_tag, "base, chua finetune")]
tasks = ["aime24", "aime25", "amc12", "math500"]

def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

print()
print("=" * 78)
print("  KET QUA  %s" % os.path.basename(model_dir))
print("=" * 78)
hdr = "  %-32s %8s %8s %8s %8s %8s"
for t, desc in runs:
    root = os.path.join("Eval", "outputs_" + t)
    summ = load(os.path.join(root, "summary.json"))
    pk = load(os.path.join(root, "pass_at_k.json"))
    if summ is None and pk is None:
        print("  [%s] %s: chua co ket qua (%s)" % (t, desc, root))
        continue
    print("  [%s] %s" % (t, desc))
    print(hdr % ("metric", *tasks, "AVG"))
    if summ is not None:
        accs = summ.get("tasks", {})
        print(hdr % ("acc (mau dau tien)",
                     *["%.1f" % accs[x] if isinstance(accs.get(x), (int, float)) else "-" for x in tasks],
                     "%.1f" % summ["average_acc"] if summ.get("average_acc") is not None else "-"))
    if pk is not None:
        by_task = {r["task"]: r for r in pk.get("tasks", [])}
        for k in pk.get("k", []):
            key = "pass@%d" % k
            avg = pk.get("average", {}).get(key)
            print(hdr % (key + " (3 mau)",
                         *["%.2f" % by_task[x][key] if x in by_task and by_task[x].get(key) is not None else "-"
                           for x in tasks],
                         "%.2f" % avg if avg is not None else "-"))
    print()
print("  Thu muc output: Eval/outputs_<tag>/<task>/<tag>/*_metrics.json  |  Eval/outputs_<tag>/summary.json")
print("=" * 78)
PY
deactivate

# =============================================================================
# [Tham khao] Selective SFT (paper): can attribution truoc, env ssft_eval cho IG
# =============================================================================
# ATTR_MODEL_DIR=/mnt/local/_models/$PROJECT/DeepSeek-R1-Distill-Qwen-7B   # paper App C.3: model sinh CoT
# source /mnt/local/uvenvs/ssft_eval/bin/activate
# bash run_pipeline.sh --offline --attr-model "$ATTR_MODEL_DIR" --gpu-attr 0 --ig-no-grad-checkpoint --ig-batch-size 2 --stages split,ig,segments
# source /mnt/local/uvenvs/ssft_train/bin/activate
# bash train.sh "${FULLSFT_ARGS[@]/--full-sft/--selective}" --data data/s1k/solutions_selected.jsonl   # cung r=16 / batch 32
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --lora-r 16 --tag qwen3_8b_sel_r16_ep3
