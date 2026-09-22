# commands.sh - chay day du Selective SFT (paper) + eval cho Qwen3-8B tren server offline.
# Chay:  cd SegmentSelectiveSFT && bash commands.sh
# Moi duong dan tuong doi ben duoi (data/, SelectiveSFT/, Eval/) tinh tu repo
# root; dong cd duoi day bao dam dieu do ke ca khi goi tu thu muc khac.
# Moi giai doan dung MOT env rieng (source dung dong roi chay tiep), khong tron:
#   ssft_eval   : attribution (stage ig + segments) + eval.sh (vllm 0.10.2 + torch 2.8.0) <- ../ssft_eval.txt
#                 (torch 2.7.1 wheel PyPI khong co kernel sm_100/B200 -> phai len 2.8.0 cu128)
#   ssft_train  : train.sh + merge_lora.py (torch 2.9 + unsloth + peft)                    <- ../ssft_train.txt
# Ca hai env deu da ho tro kien truc Qwen3 (transformers 4.57 / unsloth / vllm >= 0.8.5).
#
# Luong:  solutions_selected.jsonl (tai san tu HF, da co selected_spans_ids -> KHONG chay prep/split/ig/segments)
#         -> train.sh --selective (LoRA r16) -> merge -> eval.sh --max-tokens 4096 (chi model selective)
#         -> pass@k -> bang ket qua.  Nhanh noi suy lai (IG) o buoc [1] chi chay khi thieu file do.

cd "$(dirname "${BASH_SOURCE[0]}")"
# Dung ngay khi mot buoc loi (setup check thieu goi, IG OOM ca job, train OOM, khong co checkpoint...)
# thay vi chay tiep sang eval voi checkpoint cu.
set -eo pipefail

PROJECT=aiskylimit_new_nothingnew_2          # = @PROJECT@ trong downloads.txt
# Qwen/Qwen3-8B (downloads.txt): ban post-trained co thinking mode, 36 layer, hidden 4096,
# max_position_embeddings 40960 (> 32768 nen giu nguyen max_seq_length luc train).
MODEL_DIR=/mnt/local/_models/$PROJECT/Qwen3-8B
# Model tinh IG: paper (App C.3) luon dung R1-Distill-Qwen-7B (model sinh CoT cua s1K),
# ke ca khi model train la Qwen khac. MODEL_DIR o tren chi dung cho train/eval.
ATTR_MODEL_DIR=/mnt/local/_models/$PROJECT/DeepSeek-R1-Distill-Qwen-7B
# Snapshot baesad/s1K-1.1-deepseek-cot (downloads.txt) co san CA BA file (deu 934 mau):
#   train.jsonl               question / solution (trace R1) / answer
#   solution_segments.jsonl   + segments[] chia theo "\n\n" (segment_mode paragraph, ~240 segment/mau)
#   solutions_selected.jsonl  + selected_spans_ids[] tu IG cua lan chay truoc (R1-Distill-7B, J=20):
#                             tb 16% segment/mau duoc chon; 17 mau khong chon segment nao (IG = 0 do OOM)
#                             -> train_mask.py cho hoc 3 segment mac dinh dau/gan cuoi/cuoi.
# Da doi chieu: 3 file cung question, cung segments, dung mode paragraph -> chi symlink, khong chay
# stage nao truoc train. R1-Distill chi can tai neu xoa solutions_selected.jsonl de noi suy lai.
DATA_DIR=/mnt/local/_data/$PROJECT/s1k
# Benchmark eval tai ve dang HF dataset (downloads.txt):
#   $EVAL_DATA_ROOT/aime24  aime25  MATH-500  aimo-validation-amc
# prepare_eval_data.py chuyen thanh data/<task>/test.jsonl (question + answer).
EVAL_DATA_ROOT=/mnt/local/_data/$PROJECT

# GPU dung cho ca attribution, train lan eval (cac script export CUDA_VISIBLE_DEVICES=$GPU).
# Mot so: "1" = chi GPU 1. Eval nhieu GPU: GPU=0,1,2,3 + them --data-parallel vao EVAL_ARGS (1 task/GPU).
# IG va train luon 1 GPU (1 tien trinh); effective batch train = 1 x 32 accum = 32 khong doi.
GPU=${GPU:-1}

# Do dai sinh toi da luc eval. 4096 = nhanh, nhung trace s1K trung binh ~6-7k token nen
# nhieu cau se bi cat truoc khi ra \boxed -> acc thap hon that (run truoc: 4096 -> 8192 -> 32768
# vi ly do nay). Doi so nay thi tag tu doi theo (_4k/_8k/_32k) de khong tron output.
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-4096}
LEN_TAG="$((EVAL_MAX_TOKENS / 1024))k"

# Ten run - dung chung cho tag eval va thu muc outputs_<tag>. summary.json va pass_at_k.py
# gom MOI *_metrics.json duoi outputs_<tag>, nen moi max_tokens phai co tag rieng.
TAG=qwen3_8b_sel_r16_ep3_${LEN_TAG}          # selective SFT - run duy nhat duoc eval trong file nay

# Thu muc checkpoint theo quy uoc cua train.sh: <model>_epoch<E>_lr<LR>_len<L>[_fullsft][_lora_r<R>]
SEL_CKPT_DIR=SelectiveSFT/checkpoints/$(basename "$MODEL_DIR")_epoch3_lr5e-5_len32768_lora_r16

# =============================================================================
# [1] DATA - env ssft_eval: symlink solutions_selected.jsonl (noi suy lai IG chi khi thieu file)
# =============================================================================
source /mnt/local/uvenvs/ssft_eval/bin/activate
bash setup.sh check --for eval           # phai thay vllm 0.10.2 / torch 2.8.0 / transformers / latex2sympy
ls "$DATA_DIR" "$MODEL_DIR"
# Qwen3-8B phai co tokenizer.json (fast tokenizer): train_mask.py can offset_mapping de mask segment.
# (R1-Distill kiem o nhanh noi suy ben duoi - chi can tai model do khi phai tinh lai IG.)
[[ -f "$MODEL_DIR/tokenizer.json" ]] || { echo "Thieu $MODEL_DIR/tokenizer.json"; exit 1; }

# solution_segments.jsonl da chia san (mode paragraph, da kiem "".join(segments) == solution cho
# ca 934 mau truoc khi up) -> chi symlink, KHONG chay stage split. train.jsonl chi de tham khao.
mkdir -p data/s1k
[[ -s data/s1k/train.jsonl ]]             || ln -sf "$DATA_DIR/train.jsonl" data/s1k/train.jsonl
[[ -s data/s1k/solution_segments.jsonl ]] || ln -sf "$DATA_DIR/solution_segments.jsonl" data/s1k/solution_segments.jsonl
wc -l data/s1k/solution_segments.jsonl    # 934 dong

# IG (Integrated Gradients) tu token moi segment -> token \boxed{answer}, model R1-Distill-Qwen-7B,
# ig_steps 20 (paper 50; ~5% sai so), weight dong bang, khong grad checkpointing, 2 buoc IG/forward
# (cung cau hinh da chay on tren server truoc). Buoc nang nhat cua ca pipeline: 934 mau x 20 buoc
# forward+backward tren ca trace -> vai gio tren 1 GPU. Log: logs/ig.log, logs/segments.log.
#   - Mau OOM khong lam chet job: bi gan diem 0 va chay tiep (chi hoc 3 segment mac dinh).
#     Xem dong "CANH BAO: N mau OOM" cuoi logs/ig.log; nhieu qua thi chay lai voi --ig-batch-size 1.
#   - Bi ngat giua chung: chay lai commands.sh la tu --resume tu mau do dang (kiem 'question'
#     tung dong da ghi, khong tron hai lan chay). Muon tinh lai tu dau: xoa Attribution/processed_data/s1k/.
#   - Stage segments: diem segment = sum|IG| / sqrt(len); giu top segment den 70% tong diem,
#     bo segment co |sum IG| / sum|IG| > 0.8 -> ghi selected_spans_ids vao solutions_selected.jsonl.
# Ket qua IG cua lan chay truoc da nam trong snapshot -> mac dinh di nhanh dau (khong noi suy):
#   $DATA_DIR/solutions_selected.jsonl  -> dung thang, bo qua ca ig lan segments   (TRUONG HOP HIEN TAI)
#   $DATA_DIR/IG_compact.jsonl          -> chi chay lai stage segments (CPU, vai giay; doi duoc nguong 0.7/0.8)
#   khong co file nao                   -> noi suy lai tu dau (vai gio/GPU, can R1-Distill-7B)
# File dung lai phai tinh tren DUNG solution_segments.jsonl nay; get_important_segments.py / train_mask.py
# doi chieu so segment tung mau nen lech se bao loi, khong sai am tham.
IG_RAW=Attribution/processed_data/s1k/solution_segments_attn.jsonl
IG_COMPACT=Attribution/processed_data/s1k/IG_compact.jsonl
if [[ ! -s data/s1k/solutions_selected.jsonl && -s "$DATA_DIR/solutions_selected.jsonl" ]]; then
  ln -sf "$DATA_DIR/solutions_selected.jsonl" data/s1k/solutions_selected.jsonl
fi
if [[ ! -s "$IG_COMPACT" && -s "$DATA_DIR/IG_compact.jsonl" ]]; then
  mkdir -p "$(dirname "$IG_COMPACT")" && ln -sf "$DATA_DIR/IG_compact.jsonl" "$IG_COMPACT"
fi

if [[ -s data/s1k/solutions_selected.jsonl ]]; then
  echo "Da co data/s1k/solutions_selected.jsonl - bo qua attribution (xoa file de tinh lai)"
elif [[ -s "$IG_COMPACT" ]]; then
  echo "Da co $IG_COMPACT - chi chay stage segments"
  bash run_pipeline.sh --offline --stages segments
else
  # grad_analyze.py cung can fast tokenizer (offset_mapping) cua model tinh IG.
  ls "$ATTR_MODEL_DIR"
  [[ -f "$ATTR_MODEL_DIR/tokenizer.json" ]] || { echo "Thieu $ATTR_MODEL_DIR/tokenizer.json"; exit 1; }
  IG_RESUME_ARGS=()
  [[ -s "$IG_RAW" ]] && IG_RESUME_ARGS=(--resume)
  bash run_pipeline.sh --offline --attr-model "$ATTR_MODEL_DIR" --gpu-attr "$GPU" \
      --ig-steps 20 --ig-no-grad-checkpoint --ig-batch-size 2 \
      --stages ig,segments "${IG_RESUME_ARGS[@]}"
fi
wc -l data/s1k/solutions_selected.jsonl   # 934 dong, moi dong co them selected_spans_ids
python -c "
import json
rows = [json.loads(l) for l in open('data/s1k/solutions_selected.jsonl')]
ratio = [len(r['selected_spans_ids']) / len(r['segments']) for r in rows]
empty = sum(1 for r in rows if not r['selected_spans_ids'])
print('ty le segment duoc chon: trung binh %.3f | %d/%d mau khong chon segment nao (IG = 0, OOM?)' % (sum(ratio) / len(ratio), empty, len(rows)))
"
deactivate

# =============================================================================
# [2] TRAIN - Selective SFT (chi hoc segment duoc chon), LoRA r=16 - env ssft_train
# =============================================================================
# Cau hinh (ghi tuong minh; giong het run full-CoT SFT truoc, chi khac --selective + data):
#   --selective -> train_mask.py --mask --apply_all: label = -100 tru cac segment trong
#     selected_spans_ids + luon hoc segment dau / gan cuoi / cuoi. Mask theo offset ky tu.
#   --segment-mode paragraph PHAI khop file solution_segments.jsonl (da kiem o buoc [1]).
#   LoRA r 16 / alpha 16 / dropout 0.05, target q,k,v,o,gate,up,down (mac dinh train.sh)
#   effective batch = 1 per-device x 32 accum x 1 GPU = 32 mau/step
#   -> 934 mau / 32 = 30 step/epoch, 3 epoch = 90 step
#   AdamW betas (0.9, 0.999) eps 1e-8 (mac dinh) weight_decay 0.0
#   cosine + warmup, warmup_ratio 0.1 (HF dung LambdaLR)
#   max_seq_length 32768 (< 40960 cua Qwen3-8B, khong can RoPE scaling); gradient checkpointing bat.
#   KHONG giam --max-seq-length theo eval 4096: mau bi cat mat het segment duoc chon thi label toan
#   -100 -> train_mask.py bo mau do (bao so luong) va model hoc it hon han.
# Chat template Qwen3: apply_chat_template(add_generation_prompt=True) mac dinh (thinking mode) ket thuc
# bang "<|im_start|>assistant\n" - khop response_template cua train_mask.py voi --think-prefix none
# (train_mask.py tu probe template va dung ngay neu lech). KHONG dung --think-prefix plain/special:
# trace s1K la text tran, khong co the <think>, va prompt luc eval (math_eval.py --apply_chat_template,
# khong --enable_think) cung khong chen <think>.
# Thu muc checkpoint: $SEL_CKPT_DIR (train.sh tu ghep _lora_r16, khong co _fullsft;
# eval.sh --selective --lora-r 16 tim dung thu muc nay). Log: logs/train_lora_r16.log
# VRAM: weight bf16 ~16.4 GB + adapter r16 nho; nang nhat la logits 32768 x 152k vocab fp32 ~20 GB (+grad)
# -> ~65-80 GB, du cho 1 GPU 80 GB (mask khong doi VRAM). Neu OOM: --4bit hoac --lora-r 8.
source /mnt/local/uvenvs/ssft_train/bin/activate
bash setup.sh check --for train          # phai thay torch 2.9 / unsloth / peft / torchao<0.18

TRAIN_ARGS=(
  --offline --model "$MODEL_DIR" --gpu "$GPU"
  --selective --data data/s1k/solutions_selected.jsonl --segment-mode paragraph
  --lora-r 16 --lora-alpha 16 --lora-dropout 0.05
  --epochs 3 --lr 5e-5 --max-seq-length 32768
  --batch-size 1 --grad-accum 32
  --optim adamw_torch --weight-decay 0.0 --lr-scheduler cosine --warmup-ratio 0.1
  --think-prefix none
)
bash train.sh "${TRAIN_ARGS[@]}" --dry-run     # in lenh truoc, chua chay
bash train.sh "${TRAIN_ARGS[@]}"               # log: logs/train_lora_r16.log

# Checkpoint la adapter LoRA -> merge vao weight goc truoc khi eval (peft chi co o env train; CPU du).
# '|| true': duoi set -o pipefail, ls khong khop gi lam ca pipeline loi -> set -e giet script
# truoc khi kip in dong bao ben duoi.
CKPT=$(ls -1d "$SEL_CKPT_DIR"/checkpoint-* 2>/dev/null | grep -v -- '-merged$' | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)
[[ -n "$CKPT" ]] || { echo "Khong thay checkpoint trong $SEL_CKPT_DIR"; exit 1; }
echo "checkpoint moi nhat: $SEL_CKPT_DIR/checkpoint-$CKPT"
if [[ ! -f "$SEL_CKPT_DIR/checkpoint-$CKPT-merged/config.json" ]]; then
  (cd SelectiveSFT && HF_HUB_OFFLINE=1 python merge_lora.py \
      --adapter "../$SEL_CKPT_DIR/checkpoint-$CKPT" --base_model "$MODEL_DIR")
fi
ls "$SEL_CKPT_DIR/checkpoint-$CKPT-merged"          # phai co config.json + model*.safetensors
deactivate

# =============================================================================
# [3] EVAL - env ssft_eval (KHONG dung ssft_train), max_tokens = $EVAL_MAX_TOKENS
# =============================================================================
# Thiet lap: t=0.6, top_p=0.9, repetition_penalty=1.05, max_tokens=4096 (EVAL_MAX_TOKENS o dau file),
# k=3 mau/cau cho MOI benchmark (aime24 aime25 amc12 math500).
# amc12 = AI-MO/aimo-validation-amc (83 cau AMC12 2022-2023).
# Pass@1 = trung binh acc tren 3 mau; Pass@3 = 1 neu bat ky mau nao dung;
# AVG = trung binh cong don gian qua 4 bo (pass_at_k.py tinh, xem cuoi file).
# vLLM: max_model_len tu lay 40960 tu config Qwen3-8B; stop_token_ids <|im_end|>=151645 /
# <|endoftext|>=151643 giong Qwen2.5 (math_eval.py bat theo "qwen" trong ten).
# Chi cham model selective SFT. Ket qua 4096 KHONG so sanh truc tiep voi outputs_qwen3_8b_fullsft_r16_ep3
# (32768) cua run truoc; muon so sanh thi cham lai baseline o cung max_tokens (xem [Tham khao] cuoi file).
source /mnt/local/uvenvs/ssft_eval/bin/activate
bash setup.sh check --for eval           # phai thay vllm 0.10.2 / torch 2.8.0 / latex2sympy
python prepare_eval_data.py --data-root "$EVAL_DATA_ROOT"   # bo qua task da co test.jsonl
wc -l data/aime24/test.jsonl data/aime25/test.jsonl data/amc12/test.jsonl data/math500/test.jsonl

EVAL_ARGS=(
  --offline --gpu "$GPU"
  --tasks "aime24 aime25 amc12 math500" --n-sampling 3
  --temperature 0.6 --top-p 0.9 --repetition-penalty 1.05 --max-tokens "$EVAL_MAX_TOKENS"
)
# --selective --lora-r 16: eval.sh tim $SEL_CKPT_DIR/checkpoint-<moi nhat>, thay adapter thi tu doi
# sang ban -merged. eval.sh resumable: task da co *_metrics.json thi bo qua (--overwrite de cham lai).
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --lora-r 16 --tag "$TAG" --dry-run
BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --selective --lora-r 16 --tag "$TAG"

# Nhieu GPU: dat GPU=0,1,2,3 o dau file va them --data-parallel vao EVAL_ARGS (1 task / GPU)

# "acc" trong summary.json chi tinh MAU DAU TIEN moi cau (evaluate.py: mean_score[0]).
# Pass@1 / Pass@3 unbiased tren du 3 mau + AVG macro qua 4 bo (ghi <root>/pass_at_k.json):
(cd Eval && python pass_at_k.py "outputs_$TAG" --k 1 3)

# =============================================================================
# [4] KET QUA - in bang tong hop ra man hinh (doc lai summary.json + pass_at_k.json)
# =============================================================================
# Chay rieng buoc nay (khong train/eval lai) khi chi muon xem lai ket qua: copy khoi python
# ben duoi, truyen 3 tham so <MODEL_DIR> <TAG> <EVAL_MAX_TOKENS>.
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
    pk = load(os.path.join(root, "pass_at_k.json"))
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
# [Tham khao] So sanh voi baseline / base o cung max_tokens (khong chay trong file nay)
# =============================================================================
# Baseline full-CoT SFT (LoRA r16, da train o run truoc, checkpoint ..._fullsft_lora_r16 da merge):
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --full-sft --lora-r 16 --tag qwen3_8b_fullsft_r16_ep3_${LEN_TAG}
# Train lai baseline neu khong con checkpoint (cung r=16 / batch 32, merge nhu buoc [2]):
# bash train.sh "${TRAIN_ARGS[@]/--selective/--full-sft}" --data data/s1k/train.jsonl
# Base Qwen3-8B chua finetune (thinking mode, o 4096 token thuong chua ket thuc <think>):
# BASE_MODEL="$MODEL_DIR" bash eval.sh "${EVAL_ARGS[@]}" --base --tag qwen3_8b_base_${LEN_TAG}
# Them tag vao list runs o buoc [4] de in chung bang.
