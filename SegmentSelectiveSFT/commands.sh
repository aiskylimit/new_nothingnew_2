# commands.sh - CHI EVAL model selective SFT (Qwen3-8B, LoRA r16, 3 epoch) tren server offline,
# max_tokens 32768, roi in bang ket qua ra man hinh.
# Chay:  cd SegmentSelectiveSFT && bash commands.sh
# Moi duong dan tuong doi ben duoi (data/, SelectiveSFT/, Eval/) tinh tu repo root; dong cd duoi day
# bao dam dieu do ke ca khi goi tu thu muc khac.
#
# Tien de: checkpoint da train + merge o run truoc nam trong $SEL_CKPT_DIR/checkpoint-<step>-merged
# (train/merge can env ssft_train + peft; file nay KHONG train lai, thieu checkpoint thi dung ngay).
# Chi dung env ssft_eval (vllm 0.10.2 + torch 2.8.0) <- ../ssft_eval.txt. Cac buoc data/attribution/
# train/merge cua phien ban truoc: xem git log (commit "commands.sh: selective SFT Qwen3-8B ...").
#
# Luong:  prepare_eval_data.py -> eval.sh --max-tokens 32768 (4 benchmark x 3 mau/cau)
#         -> pass_at_k.py -> bang ket qua (acc + pass@1 + pass@3).

cd "$(dirname "${BASH_SOURCE[0]}")"
# Dung ngay khi mot buoc loi (setup check thieu goi, khong co checkpoint, vLLM OOM...)
set -eo pipefail

PROJECT=aiskylimit_new_nothingnew_2          # = @PROJECT@ trong downloads.txt
# Qwen/Qwen3-8B (downloads.txt): max_position_embeddings 40960 > 32768 -> khong can RoPE scaling.
# eval.sh --selective doc BASE_MODEL de ghep ten thu muc checkpoint (xem SEL_CKPT_DIR).
MODEL_DIR=/mnt/local/_models/$PROJECT/Qwen3-8B
# Benchmark eval tai ve dang HF dataset (downloads.txt):
#   $EVAL_DATA_ROOT/aime24  aime25  MATH-500  aimo-validation-amc
# prepare_eval_data.py chuyen thanh data/<task>/test.jsonl (question + answer).
EVAL_DATA_ROOT=/mnt/local/_data/$PROJECT

# GPU cho eval (eval.sh export CUDA_VISIBLE_DEVICES=$GPU). Mot so: "1" = chi GPU 1.
# Nhieu GPU: GPU=0,1,2,3 + them --data-parallel vao EVAL_ARGS (1 task / GPU, nhanh hon TP).
GPU=${GPU:-1}

# Do dai sinh toi da luc eval. Trace s1K trung binh ~6-7k token, nhieu cau AIME can hon 8k;
# 4096/8192 cat truoc khi ra \boxed -> acc thap hon that. 32768 = day du (cham hon ro ret,
# ~32k x 3 mau x ~800 cau). Doi so nay thi tag tu doi theo (_4k/_8k/_32k) de khong tron output.
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-32768}
LEN_TAG="$((EVAL_MAX_TOKENS / 1024))k"

# Ten run - dung chung cho tag eval va thu muc outputs_<tag>. summary.json va pass_at_k.py
# gom MOI *_metrics.json duoi outputs_<tag>, nen moi max_tokens phai co tag rieng.
TAG=qwen3_8b_sel_r16_ep3_${LEN_TAG}

# Thu muc checkpoint theo quy uoc cua train.sh: <model>_epoch<E>_lr<LR>_len<L>[_fullsft][_lora_r<R>]
SEL_CKPT_DIR=SelectiveSFT/checkpoints/$(basename "$MODEL_DIR")_epoch3_lr5e-5_len32768_lora_r16

# =============================================================================
# [1] KIEM TRA - env ssft_eval, checkpoint da merge, du lieu benchmark
# =============================================================================
source /mnt/local/uvenvs/ssft_eval/bin/activate
bash setup.sh check --for eval           # phai thay vllm 0.10.2 / torch 2.8.0 / latex2sympy
ls "$MODEL_DIR"

# Checkpoint moi nhat (theo step) phai co ban -merged: vLLM khong nap adapter LoRA truc tiep,
# con merge_lora.py can peft (chi co o env ssft_train) -> file nay khong merge, chi bao loi.
# '|| true': duoi set -o pipefail, ls khong khop gi lam ca pipeline loi -> set -e giet script
# truoc khi kip in dong bao ben duoi.
CKPT=$(ls -1d "$SEL_CKPT_DIR"/checkpoint-* 2>/dev/null | grep -v -- '-merged$' | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)
[[ -n "$CKPT" ]] || { echo "Khong thay checkpoint trong $SEL_CKPT_DIR - chua train (xem git log de lay lai buoc train)"; exit 1; }
MERGED="$SEL_CKPT_DIR/checkpoint-$CKPT-merged"
[[ -f "$MERGED/config.json" ]] || { echo "Thieu $MERGED/config.json - merge trong env ssft_train:
  source /mnt/local/uvenvs/ssft_train/bin/activate
  cd SelectiveSFT && HF_HUB_OFFLINE=1 python merge_lora.py --adapter ../$SEL_CKPT_DIR/checkpoint-$CKPT --base_model $MODEL_DIR"; exit 1; }
echo "checkpoint eval: $MERGED"
ls "$MERGED"                              # phai co config.json + model*.safetensors

python prepare_eval_data.py --data-root "$EVAL_DATA_ROOT"   # bo qua task da co test.jsonl
wc -l data/aime24/test.jsonl data/aime25/test.jsonl data/amc12/test.jsonl data/math500/test.jsonl

# =============================================================================
# [2] EVAL - max_tokens = $EVAL_MAX_TOKENS, 4 benchmark x 3 mau/cau
# =============================================================================
# Thiet lap: t=0.6, top_p=0.9, repetition_penalty=1.05, k=3 mau/cau cho MOI benchmark
# (aime24 aime25 amc12 math500). amc12 = AI-MO/aimo-validation-amc (83 cau AMC12 2022-2023).
# vLLM: max_model_len tu lay 40960 tu config Qwen3-8B (> prompt + 32768 sinh); stop_token_ids
# <|im_end|>=151645 / <|endoftext|>=151643 giong Qwen2.5 (math_eval.py bat theo "qwen" trong ten).
# Prompt: apply_chat_template mac dinh (thinking mode), khong chen <think> - khop luc train.
EVAL_ARGS=(
  --offline --gpu "$GPU"
  --tasks "aime24 aime25 amc12 math500" --n-sampling 3
  --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 --max-tokens "$EVAL_MAX_TOKENS"
)
# --selective --lora-r 16: eval.sh tim $SEL_CKPT_DIR/checkpoint-<moi nhat>, thay adapter thi tu doi
# sang ban -merged. eval.sh resumable: task da co *_metrics.json thi bo qua (--overwrite de cham lai).
# Cuoi eval.sh tu tong hop: ghi Eval/outputs_$TAG/summary.json (acc + pass@1/pass@3) va in bang tung task.
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --lora-r 16 --tag "$TAG" --dry-run
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --lora-r 16 --tag "$TAG"

# "acc" trong summary.json chi tinh MAU DAU TIEN moi cau (evaluate.py: mean_score[0]); eval.sh
# cung ghi san pass@1 / pass@3 (k = 1 va n_sampling) vao summary.json["pass_at_k"]. pass_at_k.py
# tinh lai voi k tuy y + AVG macro qua 4 bo (ghi <root>/pass_at_k.json), cung cong thuc:
(cd Eval && python pass_at_k.py "outputs_$TAG" --k 1 3)

# =============================================================================
# [3] KET QUA - in bang tong hop ra man hinh (doc lai summary.json + pass_at_k.json)
# =============================================================================
# Chay rieng buoc nay (khong eval lai) khi chi muon xem lai ket qua: copy khoi python ben duoi,
# truyen 3 tham so <MODEL_DIR> <TAG> <EVAL_MAX_TOKENS>. Moi so lam tron 2 chu so thap phan.
python - "$MODEL_DIR" "$TAG" "$EVAL_MAX_TOKENS" <<'PY'
import json, os, sys
model_dir, tag, max_tokens = sys.argv[1:4]
runs = [(tag, "selective SFT LoRA r16, 3 epoch (paper)")]
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
# [Tham khao] So sanh voi baseline / base o cung max_tokens (khong chay trong file nay)
# =============================================================================
# Baseline full-CoT SFT (LoRA r16, checkpoint ..._fullsft_lora_r16 da merge o run truoc):
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --full-sft --lora-r 16 --tag qwen3_8b_fullsft_r16_ep3_${LEN_TAG}
# Base Qwen3-8B chua finetune (thinking mode):
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --base --tag qwen3_8b_base_${LEN_TAG}
# Them tag vao list runs o buoc [3] de in chung bang.
