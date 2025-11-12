# config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Args:
    model_s: str = "Qwen/Qwen3-1.7B"
    model_t: str = "Qwen/Qwen3-0.6B"
    train_file: str = "./dataset/train-00000-of-00002.parquet"
    valid_file: str = "./dataset/validation-00000-of-00001.parquet"
    output_dir: str = "./checkpoints"
    cache_dir: str = "./precompute"
    reuse_a_layer_start: int = 0
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    lr: float = 1e-4
    epochs: int = 3
    grad_accum_steps: int = 1
    mixed_precision: str = "no"  # no, fp16, bf16
    start_idx: int = 0
    end_idx: Optional[int] = None
    # Flags
    train: bool = False
    eval: bool = False
    test: bool = False
    adapter: bool = False
    k_reuse: bool = False
    v_reuse: bool = False
    test1: bool = False
    pre: bool = False
    
device = "cuda:6"
device0 = "cuda:7"