import torch
from transformers import DynamicCache
from typing import List
from models.mlp_adapter import AdapterBank,RidgeAdapter
from utils.postprocess import *
from utils.debug import *
from torch import nn
from typing import Callable, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def load_sharded_model(model_name_or_path, dtype=torch.float16, use_4bit=False, offload_folder="/tmp/model_offload"):
    kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # 先节省 CPU 峰值
    )
    # 推荐用 fp16 做推理（若显存极紧可考虑 4-bit）
    if use_4bit:
        # 需要 bitsandbytes；并非所有模型/环境都支持
        kwargs.update(dict(load_in_4bit=True, device_map="auto"))
    else:
        kwargs.update(dict(torch_dtype=dtype, device_map="auto", offload_folder=offload_folder))

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model

def module_device(module):
    # try parameters
    for p in module.parameters(recurse=False):
        return p.device
    # fallback: try buffers
    for b in module.buffers(recurse=False):
        return b.device
    # default
    return torch.device("cpu")