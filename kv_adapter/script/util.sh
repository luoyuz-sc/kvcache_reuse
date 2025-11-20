CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  python main.py \
    --model_s Qwen/Qwen3-8B     \
    --model_t Qwen/Qwen3-1.7B    \
    --train_file ./dataset/train-00000-of-00002.parquet    --valid_file ./dataset/validation-00000-of-00001.parquet     \
    --output_dir ./checkpoints/attn_reuse     --reuse_a_layer_start 0     \
    --grad_accum_steps 1     --epochs 10     --lr 1e-2     --mixed_precision no     --train

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python main.py \
    --model_s Qwen/Qwen3-8B     \
    --model_t Qwen/Qwen3-1.7B    \
    --train_file ./dataset/train-00000-of-00002.parquet    --valid_file ./dataset/validation-00000-of-00001.parquet     \
    --output_dir ./checkpoints/attn_reuse     --reuse_a_layer_start 0     \
    --grad_accum_steps 1     --epochs 10     --lr 1e-2     --mixed_precision no     --train

