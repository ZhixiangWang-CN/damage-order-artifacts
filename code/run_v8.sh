#!/bin/bash
# Launch the addendum-8 workers on one machine.  bash run_v8.sh <spec> ...   spec = gpu:models:max_gb:budget[:tag]  (tag keeps log names unique when a GPU gets a second worker)
# server_a:  bash run_v8.sh 0:q14,gemma:34:16000 5:q14,gemma:34:16000 1:q7,mis:19:8000 2:q7,mis:19:8000 3:q7,mis:19:8000 4:q7,mis:19:8000 6:q7,mis:19:8000
# server_b:   bash run_v8.sh 0:q15,llama,phi:14:12000 1:q15,llama,phi:14:12000 2:q15,llama,phi:14:12000 4:q15,llama,phi:14:12000 5:q15,llama,phi:14:12000 6:q15,llama,phi:14:12000
cd "$(dirname "$0")"; mkdir -p logs runs
for spec in "$@"; do
  IFS=: read gpu models maxgb budget tag <<< "$spec"
  name="$(hostname -s)-g$gpu${tag:+-$tag}"
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ATLAS_TOK_BUDGET=$budget \
    setsid nohup python3 atlas_full.py worker items runs "$models" "$maxgb" "$name" > "logs/worker_$name.log" 2>&1 < /dev/null &
  echo "started $name: $models (max ${maxgb}GB, budget $budget)"
done
