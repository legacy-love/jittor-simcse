# export HF_ENDPOINT=https://hf-mirror.com
# transformers-cli download bert-base-uncased

python evaluation.py \
    --model_name_or_path ../output/unsup.bin \
    --task_set sts \
    --mode test \
    > result_unsup.txt 2>&1 &

# python evaluation.py \
#     --model_name_or_path ../output/unsup.bin \
#     --task_set sts \
#     --mode test
python evaluation.py \
    --model_name_or_path ../output/sup5.bin \
    --task_set sts \
    --mode test \
    > result.txt 2>&1 &