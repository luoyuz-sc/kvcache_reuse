
for i in {0..3};do
    file_names="debug$i.log"
    output_dir="./precompute$i"
    CUDA_VISIBLE_DEVICES=$((i+4)) python hotpot_qa_reuse.py --model_s Qwen/Qwen3-1.7B  \
       --model_t Qwen/Qwen3-0.6B     --train_file ./dataset/train-00000-of-00002.parquet  \
       --valid_file ./dataset/validation-00000-of-00001.parquet     --output_dir $output_dir    \
     --reuse_a_layer_start 0     --grad_accum_steps 5     --epochs 10     --lr 1e-4     --mixed_precision no \
         --pre --start_idx $((i * 5000))     --end_idx $((i* 5000 + 5000))  >$file_names 2>&1  &
done