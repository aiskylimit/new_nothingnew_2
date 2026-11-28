#1 +10
#alerd-log
#v3

# cd SpectralGuidedLearning
# bash phase3_alerd_7b_8b.sh
# cd ..

# ---- in kết quả phase 3 ra log (GPU 0: qwen25-7b, GPU 1: qwen3-8b, ALERD LoRA s1K). Chỉ đọc file.
cd SpectralGuidedLearning
/mnt/local/uvenvs/spectral_guided_learning/bin/python - <<'PY'
import json
from pathlib import Path

BENCH = ["aime24", "aime25", "math500", "amc12"]
ROWS = [(f"{model} {arm}", f"{name}-{model}-s1k")
        for model in ("qwen25-7b", "qwen3-8b")
        for arm, name in (("ALERD", "iwc-gain-l05-lora"), ("SFT", "sft-nll-lora"), ("zero-shot", "zeroshot"))]
print("pass@1 / pass@3 (%)")
print(f"{'run':22}" + "".join(f"  {b:>14}" for b in BENCH + ["avg"]))
for label, tag in ROWS:
    path = Path("rebuttal/results", tag, "summary.json")
    if not path.is_file():
        print(f"{label:22}  chưa có điểm ({path})")
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
    print(f"{label:22}" + "".join(f"  {c:>14}" for c in cells + [avg]))
PY
for m in qwen25-7b qwen3-8b; do
  echo; echo "== $m: đuôi log phase 3"
  log=$(ls -t rebuttal/logs/phase3-gpu*-$m-*.log 2>/dev/null | head -1)
  if [ -n "$log" ]; then echo "$log"; tail -c 20000 "$log" | tr '\r' '\n' | tail -n 15; else echo "(chưa có log)"; fi
done
cd ..

# cd ./offline_olmo_seeds_b200
# bash ./project_commands.sh

# cd ./offline_olmo_seeds_b200_rev
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
