#1 +30
#logs
#v1

# cd SpectralGuidedLearning
# bash project_commands_phase2.sh

# ---- tiến độ các run SGL (chỉ đọc file, không thay đổi gì)
cd SpectralGuidedLearning
bash runs_status.sh   # trạng thái từng run: - / train k/3 / eval / done (+ điểm)
prog() {  # <tên run> "<log glob...>": log mới nhất, dòng progress mới nhất, 2 dòng log cuối
  local f; f=$(ls -t $2 2>/dev/null | head -1)
  echo "== $1"
  [[ -n "$f" ]] || { echo "   (chưa có log)"; return; }
  echo "   $f  (sửa lúc $(date -r "$f" '+%m-%d %H:%M:%S'))"
  tail -c 300000 "$f" | tr '\r' '\n' | grep -E '[0-9]+/[0-9]+ \[' | tail -1 | sed 's/^/   /' | cut -c1-170
  tail -c 300000 "$f" | tr '\r' '\n' | grep -vE '[0-9]+/[0-9]+ \[|^\s*$' | tail -2 | sed 's/^/   /' | cut -c1-170
}
W4R=results/w4/runs/r1q1.5b__alg
prog "run 1  (GPU 0) ALERD s1K-1.1"     "rebuttal/logs/phase2-gpu0-run1-alerd-s1k-*.log"
prog "run 10 (GPU 0) answer-steps"      "rebuttal/logs/phase2-gpu0-run10-answer-steps-*.log"
prog "run 30 (GPU 1) ALG-hard H59"      "rebuttal/logs/phase2-gpu1-run30-hard-h59-*.log"
prog "run 13 (GPU 6) ASFT"              "rebuttal/logs/phase2-gpu6-run13-asft-*.log"
prog "run 15 (GPU 6) positional R6"     "rebuttal/logs/phase2-gpu6-run15-positional-*.log"
prog "run 18 (GPU 7) W4 P1/S0"          "${W4R}__P1__S0__tau1_lam0.5_c2__seed42/train/stdout.log ${W4R}__P1__S0__tau1_lam0.5_c2__seed42/eval.log"
prog "run 20 (GPU 7) W4 P0/S1"          "${W4R}__P0__S1__tau1_lam0.5_c2__seed42/train/stdout.log ${W4R}__P0__S1__tau1_lam0.5_c2__seed42/eval.log"
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
