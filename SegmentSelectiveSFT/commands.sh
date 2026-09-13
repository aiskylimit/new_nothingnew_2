source /mnt/local/uvenvs/ssft_eval/bin/activate
MODEL_DIR=/mnt/local/_models/aiskylimit_new_nothingnew_2/Qwen2.5-7B-Instruct
# Model tinh IG: paper (App C.3) luon dung R1-Distill-Qwen-7B (model sinh CoT),
# ke ca khi train Qwen2.5-7B-Instruct. MODEL_DIR o tren chi dung cho stage train.
ATTR_MODEL_DIR=/mnt/local/_models/aiskylimit_new_nothingnew_2/DeepSeek-R1-Distill-Qwen-7B
DATA_DIR=/mnt/local/_data/aiskylimit_new_nothingnew_2/s1k
ls "$DATA_DIR" "$MODEL_DIR" "$ATTR_MODEL_DIR"

mkdir -p data/s1k && ln -sf "$DATA_DIR/train.jsonl" data/s1k/train.jsonl

bash run_pipeline.sh --offline --attr-model "$ATTR_MODEL_DIR" --gpu-attr 0 --ig-no-grad-checkpoint --ig-batch-size 2 --stages split,ig,segments
