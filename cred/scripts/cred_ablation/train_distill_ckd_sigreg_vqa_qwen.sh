#!/bin/bash

# Số lượng GPU trên mỗi node (máy)
NUM_GPUS_PER_NODE=1

# Đường dẫn tới file script training của bạn
TRAIN_SCRIPT="train_ddp.py"

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name "models/FastVLM-1.5B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "vlm2vec_train/MMEB-train" \
    --subset_name "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/B3-Qwen2-2B_FastVLM-1.5B_ckd_sigreg_cls_tqwen3_8b" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --caching_dir caching/Qwen3VL_8B_cls \
    --kd_weight 1.0 \
    --kd_loss_type "ckd_sigreg_kd" \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config_emo.json" \
    --projector_lr 5e-4 \
    --student_hidden_dim 1536 \
    --teacher_hidden_dim 1536 \
    --sigreg_weight -1


EVAL_SUBSETS=(
  "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
  "ScienceQA" "VizWiz" "GQA" "TextVQA"
)



python eval_mmeb.py \
  --model_name "training/B3-Qwen2-2B_FastVLM-1.5B_ckd_sigreg_cls_tqwen3_8b/checkpoint-epoch-0" \
  --encode_output_path "./MMEB-eval_outputs_v5/B3-Qwen2-2B_FastVLM-1.5B_ckd_sigreg_cls_tqwen3_8b" \
  --lora True \
  --lora_r 64 \
  --lora_alpha 64 \
  --pooling eos \
  --model_backbone llava_qwen2 \
  --normalize True \
  --bf16 \
  --dataset_name vlm2vec_eval/MMEB-eval \
  --subset_name "${EVAL_SUBSETS[@]}" \
  --dataset_split test \
  --per_device_eval_batch_size 8 \
  --image_dir eval_images/ \
  --image_resolution "low" \
  --tgt_prefix_mod \
  --load_pretrained_lora True \
  --report_to none