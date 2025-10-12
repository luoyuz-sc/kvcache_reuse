import os
from datasets import load_dataset

p = "/home/gehao/lyz/hotpotqa/hotpot_qa"
print("exists:", os.path.exists(p))
print("isdir:", os.path.isdir(p))
if os.path.isdir(p):
    print("ls:", os.listdir(p)[:50])

# 尝试从 hub 加载（会报错若无网络）
try:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    print("loaded from hub, splits:", ds.keys())
except Exception as e:
    print("hub load failed:", repr(e))