#1 +10
#alerd
#v1

cd SpectralGuidedLearning
# bash project_commands_phase2.sh
# W4 driver (pid 25396) treo sau khi scoring xong: kill nó -> lane B tự chạy run 30 (GPU 1), 13 -> 15 (GPU 6);
# W4 runs 18, 20 chạy lại trên GPU 7. Chỉ kill khi mọi dấu hiệu khớp đúng job của mình.
p=25396
dir=/mnt/local/aiskylimit_new_nothingnew_2/SpectralGuidedLearning
cmd=$(ps -o args= -p "$p" 2>/dev/null)
cwd=$(readlink "/proc/$p/cwd" 2>/dev/null)
env=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null)
pcmd=$(ps -o args= -p "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" 2>/dev/null)
kids=$(ps -o pid=,comm= --ppid "$p" 2>/dev/null)
echo "pid=$p cmd=[$cmd] cwd=[$cwd] parent=[$pcmd]"; echo "$env" | grep -E '^(GPUS|VARIANTS)='; echo "children:"; echo "${kids:-  (none)}"
if [[ "$cmd" == "bash project_commands_w4.sh all A" && "$cwd" == "$dir" \
      && "$pcmd" == *project_commands_phase2.sh* ]] \
   && grep -qx 'GPUS=1 6' <<< "$env" && grep -qx 'VARIANTS=P1/S0 P0/S1' <<< "$env" \
   && ! awk 'NF && $2 != "tee"' <<< "$kids" | grep -q .; then
  echo "OK: kill W4 driver $p (+ its tee)"
  for k in $(ps -o pid= --ppid "$p"); do kill -9 "$k"; done
  kill -9 "$p"      # -9: không kích hoạt trap `kill 0` (trap đó sẽ giết cả phase 2)
  GPUS=7 VARIANTS="P1/S0 P0/S1" bash project_commands_w4.sh all A
else
  echo "STOP: không khớp điều kiện, không kill, không chạy W4"
fi

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
