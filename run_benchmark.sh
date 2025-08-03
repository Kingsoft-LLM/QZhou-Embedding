POOLING_MODE=mean
normalize=true
use_instruction=true
export TOKENIZERS_PARALLELISM=true

model_name_or_path=Kingsoft-LLM/QZhou-Embedding

python3 ./run_cmteb_all.py \
    --model_name_or_path ${model_name_or_path}  \
    --pooling_mode ${POOLING_MODE} \
    --normalize ${normalize} \
    --use_instruction ${use_instruction} \
    --output_dir ./results

python3 ./run_mteb_all_v2.py \
    --model_name_or_path ${model_name_or_path}  \
    --pooling_mode ${POOLING_MODE} \
    --normalize ${normalize} \
    --use_instruction ${use_instruction} \
    --output_dir ./results