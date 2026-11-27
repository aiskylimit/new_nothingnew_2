#1 +10
#alerd-log
#v1

# cd SpectralGuidedLearning
# GPUS="0 1 6" bash project_commands_tune_rebuttal.sh

# ---- in kết quả ra log: điểm run mới cạnh run cũ, bảng so sánh, đuôi log driver. Chỉ đọc file.
cd SpectralGuidedLearning
/mnt/local/uvenvs/spectral_guided_learning/bin/python - <<'PY'
import json
from pathlib import Path

BENCH = ["aime24", "aime25", "math500", "amc12"]
NEW, OLD = "rebuttal/tune/results", "rebuttal/results"
ROWS = [
    ("run 1  old: ALERD tau 1, s1K", OLD, "iwc-gain-l05-r1-qwen-1.5b-s1k"),
    ("run 1  new: ALERD tau 0.5, s1K", NEW, "iwc-gain-t05-l05-r1-qwen-1.5b-s1k"),
    ("run 10 old: answer-steps, s1K", OLD, "ansstep-mention-r1-qwen-1.5b-s1k"),
    ("run 10 new: answer-steps, 966", NEW, "ansstep-mention-r1-qwen-1.5b-palign"),
    ("run 13 old: ASFT KL 0.05, 966", OLD, "sft-asft-r1-qwen-1.5b-palign"),
    ("run 13 new: ASFT KL 0.1, 966", NEW, "sft-asft-kl01-r1-qwen-1.5b-palign"),
]
print("pass@1 / pass@3 (%)")
print(f"{'run':32}" + "".join(f"  {b:>14}" for b in BENCH + ["avg"]))
for label, root, tag in ROWS:
    path = Path(root, tag, "summary.json")
    if not path.is_file():
        print(f"{label:32}  not finished ({path} missing)")
        continue
    scores = {r["benchmark"]: r for r in json.loads(path.read_text())}
    cells, p1, p3 = [], [], []
    for b in BENCH:
        if b not in scores:
            cells.append("-")
            continue
        p1.append(100 * scores[b]["pass@1"])
        p3.append(100 * scores[b].get("pass@3", float("nan")))
        cells.append(f"{p1[-1]:.2f} / {p3[-1]:.2f}")
    avg = f"{sum(p1) / len(p1):.2f} / {sum(p3) / len(p3):.2f}" if p1 else "-"
    if p1 and len(p1) < len(BENCH):
        avg += f" ({len(p1)} bench)"
    print(f"{label:32}" + "".join(f"  {c:>14}" for c in cells + [avg]))
PY
echo; echo "== answer-steps 966: số mẫu giữ lại, tỉ lệ token được giám sát"
grep -h "kept .* samples" rebuttal/tune/logs/ansstep-mention-r1-qwen-1.5b-palign-answer_mask.log 2>/dev/null || echo "(không thấy log answer_mask)"
echo; echo "== đuôi log driver"
driver=$(ls -t rebuttal/tune/logs/driver-*.log 2>/dev/null | head -1)
if [ -n "$driver" ]; then tail -n 30 "$driver"; else echo "(chưa có log driver)"; fi
cd ..

# cd ./offline_olmo_seeds_b200
# bash ./project_commands.sh

nvidia-smi
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/talas_vlm_embed/MMEB-evaloutputs-json-v2/ +a

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

# cd ./OpenED
# bash ./project_commands_23.sh

# cd ./VLM_Embed_Gvendi
# bash ./project_commands.sh

# cd ./learn2follow
# bash ./project_commands.sh

# cd ./contra-kd-datdvt
# bash ./project_commands.sh

# cd ./cred
# bash ./project_commands.sh

# cd ./opsd
# bash ./project_commands.sh

# cd ./talas_vlm_embed
# bash ./project_commands.sh

# cd ./grit
# bash ./project_commands.sh
