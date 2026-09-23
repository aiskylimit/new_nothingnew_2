# commands.sh - Selective SFT FULL FINETUNING tren DeepSeek-R1-Distill-Qwen-1.5B (khong LoRA), roi eval,
# tren server offline.
# Chay:  cd SegmentSelectiveSFT && bash commands.sh
# Moi duong dan tuong doi ben duoi (data/, SelectiveSFT/, Eval/) tinh tu repo root; dong cd duoi day
# bao dam dieu do ke ca khi goi tu thu muc khac.
#
# Luong:  data/s1k/solutions_selected.jsonl (tai san, da co selected_spans_ids tu IG R1-Distill-7B)
#         -> train.sh --full-finetune (env ssft_train) -> checkpoint-<moi nhat> (weight day du, KHONG merge)
#         -> eval.sh --model <ckpt> (env ssft_eval) -> pass_at_k.py -> bang ket qua (acc + pass@1/pass@3).
#
# Chat template: R1-Distill dung template DeepSeek (<｜User｜>...<｜Assistant｜><think>\n), khong phai ChatML.
# train.sh tu bat --deepseek khi ten model chua "DeepSeek-R1" -> train_mask.py probe template va dung ngay neu
# khong khop. Luc eval, math_eval.py --apply_chat_template dung chinh template luu kem checkpoint nen prompt
# eval ket thuc bang "<｜Assistant｜><think>\n" giong het luc train.
# Attribution (IG) van la cua R1-Distill-7B nhu paper (upstream run_train.sh cung train 1.5B tren data
# ..._7B_J50) -> khong chay lai stage ig.

cd "$(dirname "${BASH_SOURCE[0]}")"
# Dung ngay khi mot buoc loi (setup check thieu goi, khong co checkpoint, OOM...)
set -eo pipefail

PROJECT=aiskylimit_new_nothingnew_2          # = @PROJECT@ trong downloads.txt
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (downloads.txt): max_position_embeddings 131072.
MODEL_DIR=/mnt/local/_models/$PROJECT/DeepSeek-R1-Distill-Qwen-1.5B
# baesad/s1K-1.1-deepseek-cot: train.jsonl, solution_segments.jsonl, solutions_selected.jsonl
DATA_DIR=/mnt/local/_data/$PROJECT/s1k
# Benchmark eval tai ve dang HF dataset (downloads.txt):
#   $EVAL_DATA_ROOT/aime24  aime25  MATH-500  aimo-validation-amc
# prepare_eval_data.py chuyen thanh data/<task>/test.jsonl (question + answer).
EVAL_DATA_ROOT=/mnt/local/_data/$PROJECT

# GPU cho train va eval (train.sh / eval.sh export CUDA_VISIBLE_DEVICES=$GPU). Train chi dung 1 GPU.
GPU=${GPU:-0}

# Cau hinh train - lr 3e-5 lay tu upstream SelectiveSFT/run_train.sh (paper, R1-Distill-1.5B full FT);
# 3 epoch nhu cac run khac cua fork. Doi so nao thi ten thu muc checkpoint doi theo.
EPOCHS=${EPOCHS:-3}
LR=${LR:-3e-5}
MAX_SEQ_LENGTH=32768
# SKIP_TRAIN=1: bo qua buoc train, eval checkpoint moi nhat da co (vd sau khi train xong, eval lai
# voi max_tokens khac). train.sh KHONG tu bo qua khi da co checkpoint - chay lai la train lai tu dau.
SKIP_TRAIN=${SKIP_TRAIN:-0}

# Do dai sinh toi da luc eval. R1-Distill la model long-CoT: 4096 cat mat \boxed cua phan lon cau
# -> acc thap gia. Mac dinh 32768 (= max_seq_length luc train). Tag tu doi theo (_4k/_32k).
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-32768}
LEN_TAG="$((EVAL_MAX_TOKENS / 1024))k"

# Ten run - dung chung cho tag eval va thu muc outputs_<tag>. summary.json va pass_at_k.py
# gom MOI *_metrics.json duoi outputs_<tag>, nen moi cau hinh phai co tag rieng.
TAG=r1_1p5b_sel_ft_ep${EPOCHS}_${LEN_TAG}
BASE_TAG=r1_1p5b_base_${LEN_TAG}             # base chua finetune (chi de in doi chieu, xem [Tham khao])

# Thu muc checkpoint theo quy uoc train.sh: <model>_epoch<E>_lr<LR>_len<L>[_fullsft][_lora_r<R>][...]
# Full finetune -> khong co _lora_r<R>.
CKPT_DIR=SelectiveSFT/checkpoints/$(basename "$MODEL_DIR")_epoch${EPOCHS}_lr${LR}_len${MAX_SEQ_LENGTH}

# =============================================================================
# [1] SELECTIVE SFT - env ssft_train, full finetuning
# =============================================================================
source /mnt/local/uvenvs/ssft_train/bin/activate
bash setup.sh check --for train          # phai thay torch 2.9 / unsloth / peft / torchao<0.18

mkdir -p data/s1k
[[ -f data/s1k/solutions_selected.jsonl ]] || ln -sf "$DATA_DIR/solutions_selected.jsonl" data/s1k/solutions_selected.jsonl
wc -l data/s1k/solutions_selected.jsonl  # 934 dong

# Cau hinh train (ghi tuong minh):
#   full finetuning toan bo 1.5B (unsloth full_finetuning=True), khong adapter
#   effective batch = 1 per-device x 32 accum x 1 GPU = 32 mau/step -> ~30 step/epoch, 3 epoch ~ 88 step
#   AdamW betas (0.9, 0.999) eps 1e-8 weight_decay 0.0, cosine + warmup_ratio 0.1
#   max_seq_length 32768; segment = paragraph (moi doan "\n\n" = 1 segment), selective: chi hoc
#   segment co trong selected_spans_ids (--mask --apply_all do train.sh them)
#   gradient checkpointing: bat (mac dinh)
# VRAM: 1.78B x 16 byte (weight + grad + 2 state AdamW fp32 + master) ~ 28 GB; logits 32768 x 152k vocab
# fp32 ~ 20 GB (+ grad) -> peak co the ~ 70-80 GB. Neu OOM: --optim adamw_8bit.
# KHONG giam --max-seq-length vi se cat mat response cua mau dai.
TRAIN_ARGS=(
  --offline --model "$MODEL_DIR" --gpu "$GPU"
  --full-finetune --deepseek --selective
  --data data/s1k/solutions_selected.jsonl
  --epochs "$EPOCHS" --lr "$LR" --max-seq-length "$MAX_SEQ_LENGTH"
  --batch-size 1 --grad-accum 32
  --optim adamw_torch --weight-decay 0.0 --lr-scheduler cosine --warmup-ratio 0.1
  --segment-mode paragraph
)
if [[ "$SKIP_TRAIN" != "1" ]]; then
  bash train.sh "${TRAIN_ARGS[@]}" --dry-run   # in lenh truoc, chua chay; kiem tra "template : DeepSeek R1"
  bash train.sh "${TRAIN_ARGS[@]}"             # log: logs/train.log (ghi de moi lan chay)
fi

# Full finetuning luu thang weight day du -> KHONG can merge_lora.py.
# '|| true': duoi set -o pipefail, ls khong khop gi lam ca pipeline loi -> set -e giet script
# truoc khi kip in dong bao ben duoi.
CKPT=$(ls -1d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
[[ -n "$CKPT" ]] || { echo "Khong thay checkpoint-* trong $CKPT_DIR - xem logs/train.log"; exit 1; }
MODEL_CKPT="$CKPT_DIR/checkpoint-$CKPT"
[[ -f "$MODEL_CKPT/config.json" ]] || { echo "Thieu $MODEL_CKPT/config.json"; exit 1; }
echo "checkpoint moi nhat: $MODEL_CKPT"
ls "$MODEL_CKPT"                                   # phai co config.json + model*.safetensors + tokenizer
deactivate

# =============================================================================
# [2] EVAL - env ssft_eval (KHONG dung ssft_train)
# =============================================================================
# Thiet lap: t=0.6, top_p=0.9, repetition_penalty=1.05, k=3 mau/cau cho MOI benchmark
# (aime24 aime25 amc12 math500) - giong cac run truoc de so duoc.
# amc12 = AI-MO/aimo-validation-amc (83 cau AMC12 2022-2023).
# vLLM: max_model_len tu lay 131072 tu config; stop_token_ids bat theo "qwen" trong duong dan
# (DeepSeek-R1-Distill-Qwen-...) -> 151643 = <｜end▁of▁sentence｜> cua R1-Distill, dung EOS.
source /mnt/local/uvenvs/ssft_eval/bin/activate
bash setup.sh check --for eval           # phai thay vllm / torch / latex2sympy
python prepare_eval_data.py --data-root "$EVAL_DATA_ROOT"   # bo qua task da co test.jsonl
wc -l data/aime24/test.jsonl data/aime25/test.jsonl data/amc12/test.jsonl data/math500/test.jsonl

EVAL_ARGS=(
  --offline --gpu "$GPU"
  --model "$(realpath "$MODEL_CKPT")" --tag "$TAG"
  --tasks "aime24 aime25 amc12 math500" --n-sampling 3
  --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 --max-tokens "$EVAL_MAX_TOKENS"
)
# eval.sh resumable: task da co *_metrics.json thi bo qua (--overwrite de cham lai). Cuoi eval.sh tu
# tong hop: ghi Eval/outputs_$TAG/summary.json (acc + pass@1/pass@3) va in bang.
bash eval.sh "${EVAL_ARGS[@]}" --dry-run  # in lenh truoc, chua chay
bash eval.sh "${EVAL_ARGS[@]}"

# "acc" trong summary.json chi tinh MAU DAU TIEN moi cau (evaluate.py: mean_score[0]); pass_at_k.py
# tinh pass@1 / pass@3 khong chech + AVG macro qua 4 bo (ghi <root>/pass_at_k.json):
(cd Eval && python pass_at_k.py "outputs_$TAG" --k 1 3)

# =============================================================================
# [3] KET QUA - in bang tong hop ra man hinh (doc lai summary.json + pass_at_k.json)
# =============================================================================
# In ca base chua finetune (tag $BASE_TAG) neu da eval, de doi chieu.
# Chay rieng buoc nay (khong eval lai) khi chi muon xem lai ket qua: copy khoi python ben duoi,
# truyen 4 tham so <MODEL_DIR> <TAG> <BASE_TAG> <EVAL_MAX_TOKENS>. Moi so lam tron 2 chu so thap phan.
python - "$MODEL_DIR" "$TAG" "$BASE_TAG" "$EVAL_MAX_TOKENS" <<'PY'
import json, os, sys
model_dir, tag, base_tag, max_tokens = sys.argv[1:5]
runs = [(tag, "selective SFT full finetune"),
        (base_tag, "base chua finetune")]
tasks = ["aime24", "aime25", "amc12", "math500"]

def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

print()
print("=" * 78)
print("  KET QUA  %s  (max_tokens=%s, 3 mau/cau)" % (os.path.basename(model_dir), max_tokens))
print("=" * 78)
hdr = "  %-32s %8s %8s %8s %8s %8s"
for t, desc in runs:
    root = os.path.join("Eval", "outputs_" + t)
    summ = load(os.path.join(root, "summary.json"))
    # pass@k: pass_at_k.json (pass_at_k.py, k tuy y) hoac summary.json (eval.sh tu tinh)
    pk = load(os.path.join(root, "pass_at_k.json")) or (summ or {}).get("pass_at_k")
    if summ is None and pk is None:
        print("  [%s] %s: chua co ket qua (%s)" % (t, desc, root))
        continue
    print("  [%s] %s" % (t, desc))
    print(hdr % ("metric", *tasks, "AVG"))
    if summ is not None:
        accs = summ.get("tasks", {})
        print(hdr % ("acc (mau dau tien)",
                     *["%.2f" % accs[x] if isinstance(accs.get(x), (int, float)) else "-" for x in tasks],
                     "%.2f" % summ["average_acc"] if summ.get("average_acc") is not None else "-"))
    if pk is not None:
        tk = pk.get("tasks", [])
        by_task = tk if isinstance(tk, dict) else {r["task"]: r for r in tk}   # summary: dict, pass_at_k.json: list
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
# [Tham khao] Cac lenh khac (khong chay trong file nay)
# =============================================================================
# Base R1-Distill-1.5B chua finetune, cung thiet lap eval (tag $BASE_TAG, bang [3] tu in kem):
# source /mnt/local/uvenvs/ssft_eval/bin/activate
# BASE_MODEL="$MODEL_DIR" bash eval.sh --offline --gpu "$GPU" --base --tag r1_1p5b_base_32k \
#   --tasks "aime24 aime25 amc12 math500" --n-sampling 3 --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 \
#   --max-tokens 32768
# Baseline full-CoT SFT (khong mask, checkpoint ..._fullsft, log logs/train_fullsft.log):
# bash train.sh "${TRAIN_ARGS[@]}" --full-sft
# Eval 4096 thay vi 32k (tag tu thanh ..._4k):
# SKIP_TRAIN=1 EVAL_MAX_TOKENS=4096 bash commands.sh
