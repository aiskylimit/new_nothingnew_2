source /mnt/local/uvenvs/ssft_eval/bin/activate
MODEL_DIR=/mnt/local/_models/aiskylimit_new_nothingnew_2/Qwen2.5-7B-Instruct
DATA_DIR=/mnt/local/_data/aiskylimit_new_nothingnew_2/s1k
ls "$DATA_DIR" "$MODEL_DIR"

mkdir -p data/s1k && ln -sf "$DATA_DIR/train.jsonl" data/s1k/train.jsonl

bash run_pipeline.sh --offline --attr-model "$MODEL_DIR" --gpu-attr 0 --ig-batch-size 8 --stages split,ig,segments
