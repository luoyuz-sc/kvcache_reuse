# train_adapter_accelerate.py
import os
import argparse
import re, string
from collections import Counter
from typing import Dict, List, Tuple
import math
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from accelerate import Accelerator

# ---------------- utils (from your code) ----------------
def _white_space_fix(text: str) -> str: return " ".join(text.split())
def _remove_articles(text: str) -> str: return re.sub(r"\b(a|an|the)\b", " ", text)
def _remove_punc(text: str) -> str: return text.translate(str.maketrans("", "", string.punctuation))
def normalize(text: str) -> str:
    return _white_space_fix(_remove_articles(_remove_punc(text.lower())))
def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))
def f1_score(pred: str, gold: str) -> float:
    p_tokens, g_tokens = normalize(pred).split(), normalize(gold).split()
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return float(p_tokens == g_tokens)
    common = sum((Counter(p_tokens) & Counter(g_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

def postprocess(generated: str) -> str:
    for line in generated.strip().splitlines():
        ans = line.strip()
        if not ans:
            continue
        ans = re.sub(r"^(Answer|A|Assistant|Final Answer|</think>)\s*[:\-]\s*", "", ans, flags=re.I)
        ans = ans.strip(" #*`>\"'")
        return ans
    return generated.strip()

# ---------------- sampling utils ----------------
@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[choice].item())
    else:
        choice = torch.multinomial(probs, num_samples=1)
        return int(choice.item())

# ---------------- KV Adapter ----------------
class KVAdapter(nn.Module):
    """MLP adapter for KV. Input shape expected [B, H, S, D_in] -> output [B, H, S, D_out]"""
    def __init__(self, input_size: int, hidden_size: int = None, output_size: int = None):
        super().__init__()
        hidden_size = hidden_size or input_size
        output_size = output_size or input_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, S, D]
        B, H, S, D = x.shape
        x_flat = x.view(B * H * S, D)
        out_flat = self.mlp(x_flat)
        return out_flat.view(B, H, S, -1)
    
class AdapterBank(nn.Module):
    def __init__(self, mlps: List[nn.Module]):
        super().__init__()
        self.mlps = nn.ModuleList(mlps)

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.mlps[idx](x)

# ---------------- helper: map layers & reuse ----------------
def map_layer_nearest(idx_t, n_layers_s, n_layers_t):
    if n_layers_t <= 1:
        return 0
    return int(round(idx_t * (n_layers_s - 1) / (n_layers_t - 1)))

def reuse_layer_with_mlp(pkv_t, pkv_s, mlps: AdapterBank, device, reuse_a_layer_start: int):
    """
    pkv_t: teacher past_key_values as tuple of (k,v) per layer (tensors on device)
    pkv_s: student past_key_values
    mlps: ModuleList, length == num_reused layers
    device: target device (accelerator.device)
    returns: adapted_kv_list (list of (k_adapted, v_adapted)) on `device`, and reuse_map (list indices in pkv_s)
    """
    # note: we treat pkv_t as source to adapt; mapping maps positions in student's cache to teacher layers
    n_a = len(pkv_s)
    n_b = len(pkv_t)
    reuse_map = [map_layer_nearest(i, n_a, n_b) for i in range(reuse_a_layer_start, n_a)]
    adapted = []
    for i, b_idx in enumerate(reuse_map):
        k_t, v_t = pkv_t[b_idx]
        # ensure on device and proper dtype
        k = k_t.to(device=device)
        v = v_t.to(device=device)
        k_ad = mlps(k,i)
        v_ad = mlps(v,i)
        adapted.append((k_ad, v_ad))
    return adapted, reuse_map

# ---------------- proxy loss: MSE between adapted KV and student KV ----------------
mse_loss = nn.MSELoss()

# ---------------- training function (accelerate) ----------------
def train(args):
    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")  # fp16 will use autocast in training if set
    device = accelerator.device
    print("Accelerator device:", device)

    # load tokenizer on main process
    tokenizer = AutoTokenizer.from_pretrained(args.model_s)
    tokenizer.pad_token = tokenizer.eos_token_id

    # load models (we will keep them frozen and in eval)
    # low_cpu_mem_usage reduces peak CPU memory during loading
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, low_cpu_mem_usage=True, torch_dtype=(torch.float16 if args.fp16 else None))
    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, low_cpu_mem_usage=True, torch_dtype=(torch.float16 if args.fp16 else None))

    # move models to device (each process has its own copy when launch with accelerate)
    model_t.to(device)
    model_s.to(device)
    model_t.eval(); model_s.eval()
    for p in model_t.parameters(): p.requires_grad = False
    for p in model_s.parameters(): p.requires_grad = False

    # prepare dataset
    data_files = {"training": args.train_file}
    ds = load_dataset("parquet", data_files=data_files)
    train_ds = ds["training"]

    # initialize mlps
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start = args.reuse_a_layer_start
    num_reused = N_s - start
    head_dim_t = model_t.config.head_dim
    head_dim_s = model_s.config.head_dim

    mlps = AdapterBank([
        KVAdapter(input_size=head_dim_t if args.adapt_teacher_to_student else head_dim_s,
                  hidden_size=int((head_dim_t+head_dim_s)//2),
                  output_size=head_dim_s if args.adapt_teacher_to_student else head_dim_t)
        for _ in range(num_reused)
    ]).to(device)

    optimizer = optim.Adam(mlps.parameters(), lr=args.lr)

    # Prepare dataloader (simple, batch_size=1 for sequence-level)
    def collate_fn(batch):
        return batch  # we process examples one-by-one
    loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Use accelerator to prepare
    mlps, optimizer, loader = accelerator.prepare(mlps, optimizer, loader)

    # training loop
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        running_loss = 0.0
        for batch in progress:
            example = batch[0]
            # build prompt tokens (adapt this if you use a chat template)
            ctx = ""
            if "context" in example:
                titles = example["context"]["title"]
                sents = example["context"]["sentences"]
                sections = [f"- {t}: {ss}" for t, ss in zip(titles, sents)]
                ctx = "\n".join(sections)
            prompt = f"You are a precise QA assistant.\nCONTEXT:\n{ctx}\nQUESTION: {example['question']}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            # forward frozen student & teacher to get past_key_values (no grad)
            with torch.no_grad():
                s_out = model_s(input_ids=input_ids, use_cache=True, output_attentions=False, output_hidden_states=False)
                t_out = model_t(input_ids=input_ids, use_cache=True, output_attentions=False, output_hidden_states=False)

            pkv_s = tuple(tuple(t for t in layer) for layer in s_out.past_key_values)
            pkv_t = tuple(tuple(t for t in layer) for layer in t_out.past_key_values)

            # apply mlp to adapt teacher->student or student->teacher depending on flag
            adapted_kv_list, reuse_map = reuse_layer_with_mlp(pkv_t, pkv_s, mlps, device, reuse_a_layer_start=start)

            # compute proxy loss: compare adapted_kv to the student kv (supervise adapted to match student)
            # ensure student kv entries chosen correspond to reuse_map
            # Note: pkv_s[reuse_idx] is (k_s, v_s)
            student_targets = [pkv_s[idx] for idx in reuse_map]  # list of (k_s, v_s) on device? move if needed
            # move targets to device and dtype
            total_loss = torch.tensor(0.0, device=device)
            cnt = 0
            for (k_ad, v_ad), (k_s, v_s) in zip(adapted_kv_list, student_targets):
                k_s_t = k_s.to(device=device, dtype=k_ad.dtype)
                v_s_t = v_s.to(device=device, dtype=v_ad.dtype)
                total_loss = total_loss + mse_loss(k_ad, k_s_t) + mse_loss(v_ad, v_s_t)
                cnt += 2
            if cnt > 0:
                total_loss = total_loss / cnt

            # backward & step using accelerator
            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += total_loss.item()
            if accelerator.is_main_process:
                progress.set_postfix({"loss": running_loss / (progress.n + 1)})

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} finished, avg loss: {running_loss / len(loader):.6f}")

        # save adapter weights (only main process)
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(mlps.state_dict(), os.path.join(args.output_dir, f"mlps_epoch{epoch+1}.pt"))

    # return final mlps (on main device)
    if accelerator.is_main_process:
        print("Training done. Adapter saved.")
    return mlps

# ---------------- argparse and entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_t", type=str, required=True)
    parser.add_argument("--model_s", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./adapter_out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reuse_a_layer_start", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", help="use fp16 training where possible")
    parser.add_argument("--adapt_teacher_to_student", action="store_true", help="adapter direction flag (teacher->student)")
    args = parser.parse_args()

    os.environ["HF_HOME"] = "/home/gehao/lyz/data/hf-cache"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


    train(args)
