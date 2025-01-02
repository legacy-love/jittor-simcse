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
    --model_name_or_path ../output/sup_t005_penalty4.bin \
    --task_set sts \
    --mode test \
    > result_sup_t005_penalty4.txt 2>&1 &

python evaluation.py \
    --model_name_or_path ../output/sup_t005_penalty5.bin \
    --task_set sts \
    --mode test \
    > result_sup_t005_penalty5.txt 2>&1 &

python evaluation.py \
    --model_name_or_path ../output/sup_t005_penalty6.bin \
    --task_set sts \
    --mode test \
    > result_sup_t005_penalty6.txt 2>&1 &

python evaluation.py \
    --model_name_or_path ../output/unsup_p01_t005_penalty05_dynamic.bin \
    --task_set sts \
    --mode test \
    > result_unsup_p01_t005_penalty05_dynamic.txt 2>&1 &