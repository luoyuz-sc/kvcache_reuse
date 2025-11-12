# main.py - KV Bridged Training with Accelerate
import os
os.environ["HF_HOME"] = "/home/gehao/lyz/data/hf-cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import re
import string
import math
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json
import gc
import numpy as np
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
)

data_files = {"valid":"./dataset/dev.parquet"}
ds = load_dataset("parquet", data_files=data_files)["valid"]
pbar =tqdm(ds)
for i,ex in enumerate(pbar):
    if i<10:
        for k in ex:
            print(k)
            print(ex[k])