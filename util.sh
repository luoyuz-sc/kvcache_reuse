python hotpot_qa_test.py --model_t Qwen/Qwen3-1.7B --model_s Qwen/Qwen3-0.6B \
    --train_file /home/gehao/lyz/train-00000-of-00002.parquet \
    --valid_file /home/gehao/lyz/validation-00000-of-00001.parquet --epochs 3 --lr 1e-4 >debug.log

accelerate launch  --num_processes 8 hotpot_qa_test.py --model_t Qwen/Qwen3-1.7B \
    --model_s Qwen/Qwen3-0.6B --train_file /home/gehao/lyz/train-00000-of-00002.parquet \
    --valid_file /home/gehao/lyz/validation-00000-of-00001.parquet --epochs 3 --lr 1e-4 >debug2.log 2>&1 

accelerate launch  --num_processes 1 hotpot_qa_test.py --model_t Qwen/Qwen3-1.7B \
    --model_s Qwen/Qwen3-0.6B --train_file /home/gehao/lyz/train-00000-of-00002.parquet \
    --valid_file /home/gehao/lyz/validation-00000-of-00001.parquet --epochs 3 --lr 1e-4

python hotpot_qa_test.py --model_t Qwen/Qwen3-1.7B --model_s Qwen/Qwen3-0.6B \
    --train_file /home/gehao/lyz/train-00000-of-00002.parquet \
    --valid_file /home/gehao/lyz/validation-00000-of-00001.parquet --epochs 3 --lr 1e-4 --eval >debug2.log