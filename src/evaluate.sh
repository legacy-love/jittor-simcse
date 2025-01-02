python evaluation.py \
    --model_name_or_path ../output/sup_p01_t005_penalty1.bin \
    --task_set sts \
    --mode test \
    > ../results/result_sup_p01_t005_penalty1.txt 2>&1 &

python evaluation.py \
    --model_name_or_path ../output/unsup_p01_t005.bin \
    --task_set sts \
    --mode test \
    > ../results/result_unsup_p01_t005.txt 2>&1 &