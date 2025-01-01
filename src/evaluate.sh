# export HF_ENDPOINT=https://hf-mirror.com
# transformers-cli download bert-base-uncased

python evaluation.py \
    --model_name_or_path ../output/unsup_p001_t0001.bin \
    --task_set sts \
    --mode test \
    > result_unsup.txt 2>&1 &

# python evaluation.py \
#     --model_name_or_path ../output/unsup.bin \
#     --task_set sts \
#     --mode test
python evaluation.py \
    --model_name_or_path ../output/sup_cls.bin \
    --task_set sts \
    --mode test \
    > result.txt 2>&1 &


python evaluation.py \
    --model_name_or_path ../output/unsup_p01_t005_penalty1.bin \
    --task_set sts \
    --mode test \
    > result_unsup_p01_t005_penalty1.txt 2>&1 &


python evaluation.py \
    --model_name_or_path ../output/sup_t005_penalty3.bin \
    --task_set sts \
    --mode test \
    > result_sup_t005_penalty3.txt 2>&1 &