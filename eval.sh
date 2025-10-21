for i in {28..28..3}
do
  echo "Starting evaluation with reuse_a_layer_start = $i"
  python hotpot_qa_reuse.py \
    --model_s Qwen/Qwen3-1.7B \
    --model_t Qwen/Qwen3-0.6B \
    --train_file ./dataset/train-00000-of-00002.parquet \
    --valid_file ./dataset/validation-00000-of-00001.parquet  \
    --output_dir ./checkpoints \
    --reuse_a_layer_start 28 \
    --grad_accum_steps 5 \
    --epochs 10 \
    --lr 1e-2 \
    --mixed_precision no \
    --eval \
    --adapter
done