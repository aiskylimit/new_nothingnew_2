# commands.sh - CHI EVAL checkpoint Selective SFT da co (Qwen3-8B, LoRA r16, 3 epoch) voi THINKING TAT,
# max_tokens 4096, tren server offline. Khong train lai.
# Chay:  cd SegmentSelectiveSFT && bash commands.sh
# Moi duong dan tuong doi ben duoi (data/, SelectiveSFT/, Eval/) tinh tu repo root; dong cd duoi day
# bao dam dieu do ke ca khi goi tu thu muc khac.
#
# THINKING TAT luc eval: eval.sh --no-think -> math_eval.py --disable_think ->
# apply_chat_template(enable_thinking=False): prompt ket thuc bang "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# nen model khong tu mo <think> nua, chi sinh mot luong duy nhat roi \boxed{}.
# LUU Y: checkpoint nay (..._lora_r16, khong _nothink) duoc train voi template mac dinh (--think_prefix none,
# khong chen khoi think), nen prompt luc eval KHAC prompt luc train o dung khoi think rong do. Day la eval
# lech dieu kien co chu y - de so voi run cu cung checkpoint, cung 4096 nhung thinking mac dinh
# (outputs_qwen3_8b_sel_r16_ep3_4k). Muon khop hoan toan thi train lai voi train.sh --think-prefix off.
# Chi dung env ssft_eval (vllm 0.10.2 + torch 2.8.0) <- ../ssft_eval.txt; KHONG dung ssft_train.
#
# Luong:  checkpoint-<moi nhat>-merged (da merge tu run truoc) -> eval.sh --no-think --max-tokens 4096
#         -> pass_at_k.py -> bang ket qua (acc + pass@1 + pass@3), in kem run cu de doi chieu.

cd "$(dirname "${BASH_SOURCE[0]}")"
# Dung ngay khi mot buoc loi (setup check thieu goi, khong co checkpoint, vLLM OOM...)
set -eo pipefail

PROJECT=aiskylimit_new_nothingnew_2          # = @PROJECT@ trong downloads.txt
# Qwen/Qwen3-8B (downloads.txt): max_position_embeddings 40960.
MODEL_DIR=/mnt/local/_models/$PROJECT/Qwen3-8B
# Benchmark eval tai ve dang HF dataset (downloads.txt):
#   $EVAL_DATA_ROOT/aime24  aime25  MATH-500  aimo-validation-amc
# prepare_eval_data.py chuyen thanh data/<task>/test.jsonl (question + answer).
EVAL_DATA_ROOT=/mnt/local/_data/$PROJECT

# GPU cho eval (eval.sh export CUDA_VISIBLE_DEVICES=$GPU). Mot so: "1" = chi GPU 1.
# Nhieu GPU: GPU=0,1,2,3 + them --data-parallel vao EVAL_ARGS (1 task / GPU, nhanh hon TP).
GPU=${GPU:-1}

# Do dai sinh toi da luc eval. 4096 la ngan so voi trace s1K (trung binh ~6-7k token): cau nao chua
# ra \boxed thi tinh sai -> acc thap hon that, nhung nhanh va so duoc voi run cu cung 4096.
# Doi so nay thi tag tu doi theo (_4k/_8k/_32k) de khong tron output.
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-4096}
LEN_TAG="$((EVAL_MAX_TOKENS / 1024))k"

# Ten run - dung chung cho tag eval va thu muc outputs_<tag>. summary.json va pass_at_k.py
# gom MOI *_metrics.json duoi outputs_<tag>, nen moi (max_tokens, thinking) phai co tag rieng.
TAG=qwen3_8b_sel_r16_ep3_nothink_${LEN_TAG}
OLD_TAG=qwen3_8b_sel_r16_ep3_${LEN_TAG}      # run cu, cung checkpoint, thinking mac dinh (chi de in doi chieu)

# Thu muc checkpoint cua run truoc theo quy uoc train.sh: <model>_epoch<E>_lr<LR>_len<L>[_fullsft][_lora_r<R>]
# (khong co _nothink vi train voi template mac dinh). Truyen thang --model <duong dan merged> cho eval.sh
# vi eval.sh --selective --no-think se tim thu muc ..._nothink (khong ton tai).
SEL_CKPT_DIR=SelectiveSFT/checkpoints/$(basename "$MODEL_DIR")_epoch3_lr5e-5_len32768_lora_r16

# =============================================================================
# [1] CHECKPOINT - lay ban -merged moi nhat cua run truoc (khong train, khong merge)
# =============================================================================
# '|| true': duoi set -o pipefail, ls khong khop gi lam ca pipeline loi -> set -e giet script
# truoc khi kip in dong bao ben duoi.
CKPT=$(ls -1d "$SEL_CKPT_DIR"/checkpoint-*-merged 2>/dev/null | sed 's#.*/checkpoint-##; s#-merged$##' | sort -n | tail -1 || true)
[[ -n "$CKPT" ]] || { echo "Khong thay checkpoint-*-merged trong $SEL_CKPT_DIR - merge o env ssft_train:"; \
  echo "  cd SelectiveSFT && python merge_lora.py --adapter ../$SEL_CKPT_DIR/checkpoint-<step> --base_model $MODEL_DIR"; exit 1; }
MERGED="$SEL_CKPT_DIR/checkpoint-$CKPT-merged"
[[ -f "$MERGED/config.json" ]] || { echo "Thieu $MERGED/config.json"; exit 1; }
echo "checkpoint eval: $MERGED"
ls "$MERGED"                                        # phai co config.json + model*.safetensors

# =============================================================================
# [2] EVAL - env ssft_eval, thinking tat, max_tokens = $EVAL_MAX_TOKENS
# =============================================================================
# Thiet lap: t=0.6, top_p=0.9, repetition_penalty=1.05, k=3 mau/cau cho MOI benchmark
# (aime24 aime25 amc12 math500) - giu y het run cu de chi khac moi thinking.
# amc12 = AI-MO/aimo-validation-amc (83 cau AMC12 2022-2023).
# vLLM: max_model_len tu lay 40960 tu config Qwen3-8B; stop_token_ids <|im_end|>=151645 /
# <|endoftext|>=151643 giong Qwen2.5 (math_eval.py bat theo "qwen" trong ten).
source /mnt/local/uvenvs/ssft_eval/bin/activate
bash setup.sh check --for eval           # phai thay vllm 0.10.2 / torch 2.8.0 / latex2sympy
python prepare_eval_data.py --data-root "$EVAL_DATA_ROOT"   # bo qua task da co test.jsonl
wc -l data/aime24/test.jsonl data/aime25/test.jsonl data/amc12/test.jsonl data/math500/test.jsonl

EVAL_ARGS=(
  --offline --gpu "$GPU" --no-think
  --model "$(realpath "$MERGED")" --tag "$TAG"
  --tasks "aime24 aime25 amc12 math500" --n-sampling 3
  --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 --max-tokens "$EVAL_MAX_TOKENS"
)
# eval.sh resumable: task da co *_metrics.json thi bo qua (--overwrite de cham lai). Cuoi eval.sh tu
# tong hop: ghi Eval/outputs_$TAG/summary.json (acc + pass@1/pass@3, "enable_thinking": false) va in bang.
bash eval.sh "${EVAL_ARGS[@]}" --dry-run  # in lenh truoc, chua chay; kiem tra dong "thinking : OFF"
bash eval.sh "${EVAL_ARGS[@]}"

# "acc" trong summary.json chi tinh MAU DAU TIEN moi cau (evaluate.py: mean_score[0]); eval.sh
# cung ghi san pass@1 / pass@3 (k = 1 va n_sampling) vao summary.json["pass_at_k"]. pass_at_k.py
# tinh lai voi k tuy y + AVG macro qua 4 bo (ghi <root>/pass_at_k.json), cung cong thuc:
(cd Eval && python pass_at_k.py "outputs_$TAG" --k 1 3)

# =============================================================================
# [3] KET QUA - in bang tong hop ra man hinh (doc lai summary.json + pass_at_k.json)
# =============================================================================
# In ca run cu (thinking mac dinh, cung checkpoint + max_tokens) neu co, de doi chieu.
# Chay rieng buoc nay (khong eval lai) khi chi muon xem lai ket qua: copy khoi python ben duoi,
# truyen 4 tham so <MODEL_DIR> <TAG> <OLD_TAG> <EVAL_MAX_TOKENS>. Moi so lam tron 2 chu so thap phan.
python - "$MODEL_DIR" "$TAG" "$OLD_TAG" "$EVAL_MAX_TOKENS" <<'PY'
import json, os, sys
model_dir, tag, old_tag, max_tokens = sys.argv[1:5]
runs = [(tag, "selective SFT LoRA r16, 3 epoch, thinking OFF (enable_thinking=False)"),
        (old_tag, "cung checkpoint, thinking mac dinh (run cu)")]
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
    # pass@k: summary.json (eval.sh tu tinh) hoac pass_at_k.json (pass_at_k.py, cho k tuy y)
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
# Eval 32k thay vi 4096 (tag tu thanh ..._nothink_32k):
# EVAL_MAX_TOKENS=32768 bash commands.sh
# Base Qwen3-8B chua finetune, thinking tat (non-thinking mode chinh thuc cua Qwen3):
# BASE_MODEL="$MODEL_DIR" bash eval.sh --offline --gpu "$GPU" --no-think --base --tag qwen3_8b_base_nothink_${LEN_TAG} \
#   --tasks "aime24 aime25 amc12 math500" --n-sampling 3 --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 \
#   --max-tokens "$EVAL_MAX_TOKENS"
# Train lai voi thinking tat de train/eval khop hoan toan (env ssft_train; checkpoint ..._lora_r16_nothink):
# bash train.sh --offline --model "$MODEL_DIR" --gpu "$GPU" --selective --data data/s1k/solutions_selected.jsonl \
#   --segment-mode paragraph --lora-r 16 --lora-alpha 16 --lora-dropout 0.05 --epochs 3 --lr 5e-5 \
#   --max-seq-length 32768 --batch-size 1 --grad-accum 32 --think-prefix off
# roi merge va: BASE_MODEL="$MODEL_DIR" bash eval.sh --no-think --selective --lora-r 16 ... (eval.sh tu tim ..._nothink)
