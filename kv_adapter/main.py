import argparse
from config import Args
import os

os.environ["HF_HOME"] = "/home/gehao/lyz/data/hf-cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from train import *
from eval import *
from test.test import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_t", type=str, required=True)
    parser.add_argument("--model_s", type=str, required=True)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--reuse_a_layer_start", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--res_train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_dis", action="store_true")
    parser.add_argument("--attn_eval", action="store_true")
    parser.add_argument("--res_eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_attn", action="store_true")
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--k_reuse", action="store_true")
    parser.add_argument("--v_reuse", action="store_true")
    parser.add_argument("--test1", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="./precompute")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test:
        test(args)
    elif args.test1:
        test1(args)
    elif args.test_attn:
        test_attn_fuser(args)
    elif args.res_train:
        train_conditional_adapter_perhead_cnn(args)
    else:
        if args.train:
            train_attn_fuser(args)
        elif args.res_eval:
            evaluate_res(args)
        elif args.attn_eval:
            evaluate_attn(args)
        elif args.eval:
            #eval_wo_reuse(args)
            evaluate(args, adapter=args.adapter)
        elif args.eval_dis:
            evaluate_attn_distr(args)