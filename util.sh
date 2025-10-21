python hotpot_qa_test.py --model_t Qwen/Qwen3-1.7B --model_s Qwen/Qwen3-0.6B \
    --train_file /home/gehao/lyz/train-00000-of-00002.parquet \
    --valid_file /home/gehao/lyz/validation-00000-of-00001.parquet --epochs 3 --lr 1e-4 >debug.log

accelerate launch  --num_processes 8 hotpot_qa_test.py --model_s Qwen/Qwen3-1.7B \
    --model_t Qwen/Qwen3-0.6B --train_file /home/gehao/lyz/train-00000-of-00002.parquet \
    --valid_file /home/gehao/lyz/validation-00000-of-00001.parquet --epochs 3 --lr 1e-4 >log.debug2.log 2>&1 

accelerate launch  --num_processes 1 hotpot_qa_test.py --model_t Qwen/Qwen3-1.7B \
    --model_s Qwen/Qwen3-0.6B --train_file /home/gehao/lyz/train-00000-of-00002.parquet \
    --valid_file /home/gehao/lyz/validation-00000-of-00001.parquet --epochs 3 --lr 1e-4

python hotpot_qa_test.py --model_t Qwen/Qwen3-1.7B --model_s Qwen/Qwen3-0.6B \
    --train_file /home/gehao/lyz/train-00000-of-00002.parquet \
    --valid_file /home/gehao/lyz/validation-00000-of-00001.parquet --epochs 3 --lr 1e-4 --eval >log.debug2.log

accelerate launch --num_processes=2 hotpot_qa_reuse.py \
    --model_s Qwen/Qwen3-1.7B \
    --model_t Qwen/Qwen3-0.6B \
    --train_file ./dataset/train-00000-of-00002.parquet \
    --output_dir ./checkpoints \
    --reuse_a_layer_start 0 \
    --grad_accum_steps 5 \
    --epochs 50 \
    --lr 1e-4 \
    --mixed_precision no \
    --train

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
    --eval

python hotpot_qa_reuse.py \
    --model_s Qwen/Qwen3-1.7B \
    --model_t Qwen/Qwen3-0.6B \
    --train_file ./dataset/train-00000-of-00002.parquet \
    --valid_file ./dataset/validation-00000-of-00001.parquet \
    --output_dir ./checkpoints \
    --reuse_a_layer_start 0 \
    --grad_accum_steps 5 \
    --epochs 10 \
    --lr 1e-4 \
    --mixed_precision no \
    --train 

python hotpot_qa_reuse.py \
    --model_s Qwen/Qwen3-1.7B \
    --model_t Qwen/Qwen3-0.6B \
    --train_file ./dataset/train-00000-of-00002.parquet \
    --valid_file ./dataset/validation-00000-of-00001.parquet \
    --output_dir ./checkpoints \
    --reuse_a_layer_start 0 \
    --grad_accum_steps 5 \
    --epochs 3 \
    --lr 1e-2 \
    --mixed_precision no \
    --test