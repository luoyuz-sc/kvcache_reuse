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

def handler(sig, frame):
    print("\n🔴 Ctrl+C detected, terminating all processes...")
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def postprocess(generated: str) -> str:
    # Take only the first non-empty line; trim common prefixes.
    for line in generated.strip().splitlines():
        ans = line.strip()
        if not ans:
            continue
        ans = re.sub(r"^(Answer|A|Assistant|Final Answer|</think>)\s*[:\-]\s*", "", ans, flags=re.I)
        ans = ans.strip(" #*`>\"'")
        return ans
    return generated.strip()


# ---------------- Generation ----------------
@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """
    logits: [vocab_size] (already the last time step)
    returns: int token id
    """
    if temperature <= 0.0:
        idx = int(torch.argmax(logits, dim=-1).item())
        return idx
        
    # Temperature
    logits = logits / temperature

    # Nucleus (top-p)
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

@torch.no_grad()
def kv_bridged_generate(model_t,model_s, tok, k_mlps, v_mlps, input_ids_list: list[int], args):
    """return: generated string"""

    eos_id = tok.eos_token_id
    max_new = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p

    input_ids = torch.tensor([input_ids_list], device=device)

    # prefill with s
    s_out = model_s(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    )
    t_out = model_t(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    )
    
    pkv_s = tuple(tuple(t for t in layer) for layer in s_out.past_key_values)
    pkv_t = tuple(tuple(t for t in layer) for layer in t_out.past_key_values)
    
    # substitue layer_a of a_cache with layer_b of b_cache
    new_a_cache, reuse_b_layer_list = reuse_layer_with_mlp(pkv_t, pkv_s, k_mlps,v_mlps, args)
    
    first_token = sample_next_token(s_out.logits[:,-1,:].squeeze(0), temperature, top_p)
    
    past = DynamicCache.from_legacy_cache(past_key_values=new_a_cache)
    generated = [first_token]
    last_token = torch.tensor([[first_token]], device=device)
    for _ in range(max_new-1):
        out = model_t(
            input_ids=last_token,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        logits = out.logits[:, -1, :].squeeze(0)  # [vocab]
        past = out.past_key_values              # now past is from A going forward

        next_id = sample_next_token(logits, temperature, top_p)
        generated.append(next_id)
        
        if next_id == eos_id:
            break
        # Prepare inputs for next step
        last_token = torch.tensor([[next_id]], device=device)
    
    text = tok.decode([t for t in generated if t != eos_id], skip_special_tokens=True)
    if ":" in text :
        pos = text.index(":")
        text=text[pos+1:].strip()
    return postprocess(text)

def pure_generate(model, tok, input_ids_list: list[int], args):
    """return: generated string"""

    eos_id = tok.eos_token_id
    max_new = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p

    input_ids = torch.tensor([input_ids_list], device=device)

    out = model(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    )
    
    past = out.past_key_values
    first_token = sample_next_token(out.logits[:,-1,:].squeeze(0), temperature, top_p)
    
    generated = [first_token]
    last_token = torch.tensor([[first_token]], device=device)
    for _ in range(max_new-1):
        out = model(
            input_ids=last_token,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        logits = out.logits[:, -1, :].squeeze(0)  # [vocab]
        past = out.past_key_values              # now past is from A going forward

        next_id = sample_next_token(logits, temperature, top_p)
        generated.append(next_id)
        
        if next_id == eos_id:
            break
        # Prepare inputs for next step
        last_token = torch.tensor([[next_id]], device=device)
    
    text = tok.decode([t for t in generated if t != eos_id], skip_special_tokens=True)
    if ":" in text :
        pos = text.index(":")
        text=text[pos+1:].strip()
    return postprocess(text)

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


def format_context(example: Dict) -> str:
    """Compact, readable multi-hop context."""
    titles = example["context"]["title"]
    sents = example["context"]["sentences"]
    sections = []
    for t, ss in list(zip(titles, sents)):
        sections.append(f"- {t}: {ss}")
    return "\n".join(sections)

INSTRUCT_HEADER = (
    "You are a precise question answering assistant. Use the CONTEXT to answer the QUESTION.\n"
    "Return the **shortest** possible answer begin with 'Answer: ' (e.g., single entity or 'yes'/'no'); no explanation.\n"
)

def get_input_ids_list(tokenizer, example: Dict) -> list[int]:
    if isinstance(example, str):
        sys = INSTRUCT_HEADER.strip()
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"QUESTION: {example}\n"}
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        return tokenizer(prompt_str).input_ids
    ctx = format_context(example)
    q = example["question"]
    sys = INSTRUCT_HEADER.strip()
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {q}\n"}
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    return tokenizer(prompt_str).input_ids

def get_answer_ids_list(tokenizer, example: Dict) -> list[str]:
    a = example["answer"]
    ans_str = f"Answer: {a}"
    return tokenizer(ans_str).input_ids

def eval_one(example: Dict, tok, model_t, model_s, k_mlps,v_mlps, args) -> Tuple[str, float, float, str]:
    input_ids_list = get_input_ids_list(tok, example)
    pred = kv_bridged_generate(model_t,model_s, tok, k_mlps,v_mlps, input_ids_list, args)
    gold = example["answer"]
    with open(f"./log/qa_result.log", "a") as f:
        f.write(f"Q: {example['question']}\nA: {pred}\nG: {gold}\n")
    return pred, exact_match(pred, gold), f1_score(pred, gold)

# ---------------- MLP Adapter ----------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / norm * self.weight).to(x_dtype)

class KAdapter(nn.Module):
    """Simple MLP to adapt teacher KV to align better with student expectations."""
    def __init__(self, input_size: int, hidden_size: int = None, output_size: int = None):
        super().__init__()
        hidden_size = hidden_size or input_size
        output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.norm = RMSNorm(output_size)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, num_heads, head_dim]
        B, H, S, D = x.shape
        x_flat = x.view(B * S * H, D)
        out_flat = self.norm(self.mlp(x_flat))
        return out_flat.view(B, H, S, self.output_size)
    
class KVAdapter(nn.Module):
    """Simple MLP to adapt teacher KV to align better with student expectations."""
    def __init__(self, input_size: int, hidden_size: int = None, output_size: int = None):
        super().__init__()
        hidden_size = hidden_size or input_size
        output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
            RMSNorm(output_size)
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, num_heads, head_dim]
        B, H, S, D = x.shape
        x_flat = x.view(B * S * H, D)
        out_flat = self.mlp(x_flat)
        return out_flat.view(B, H, S, self.output_size)


class VAdapter(nn.Module):
    """Simple MLP to adapt teacher KV to align better with student expectations."""
    def __init__(self, input_size: int, hidden_size: int = None, output_size: int = None):
        super().__init__()
        hidden_size = hidden_size or input_size
        output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, num_heads, head_dim]
        B, H, S, D = x.shape
        x_flat = x.view(B * S * H, D)
        out_flat = self.mlp(x_flat)
        return out_flat.view(B, H, S, self.output_size)

    
class AdapterBank(nn.Module):
    def __init__(self, mlps: List[nn.Module]):
        super().__init__()
        self.mlps = nn.ModuleList(mlps)
    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.mlps[idx](x)

# ---------------- Modified Reuse Layer with MLP ----------------
def reuse_layer_with_mlp(a_cache, b_cache, kmlps: AdapterBank, vmlps: AdapterBank, args):
    """
    a_cache: [N_a, 2, (B, H, S, D)]  ← target KV cache
    b_cache: [N_b, 2, (B, H, S, D)]  ← source KV cache
    mlps: ModuleList of K MLPs, one for each reused layer
    Transforms selected b_cache KV states using MLP and injects into a_cache.
    """

    reuse_a_layer_start = args.reuse_a_layer_start

    reuse_b_layer_list = [
        map_layer_nearest(layer_idx, len(a_cache), len(b_cache))
        for layer_idx in range(reuse_a_layer_start, len(a_cache))
    ]

    adapted_kv_list = []
    for i, b_idx in enumerate(reuse_b_layer_list):
        k,v = b_cache[b_idx]  
        # Apply MLP adapter
        if kmlps is not None:
            k = kmlps(k, b_idx)
        if vmlps is not None:
            v = vmlps(v, b_idx)
        adapted_kv_list.append((k,v))

    # Replace from `reuse_a_layer_start` onward
    new_a_cache = list(a_cache[:reuse_a_layer_start]) + adapted_kv_list
    return tuple(new_a_cache), reuse_b_layer_list
    #return tuple(a_cache), reuse_b_layer_list

def map_layer_nearest(idx_t, n_layers_s, n_layers_t):
    return idx_t

class HotPotQADataset(Dataset):
    def __init__(self, data_files, num=1000, split="train"):
        ds = load_dataset("parquet", data_files=data_files)[split]
        self.examples = list(ds)[:num]
    
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

# ---------------- Training Loop ----------------
def train_distr(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True)
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True)
    model_s.eval().requires_grad_(False)
    model_t.eval().requires_grad_(False)

    # Build MLP adapters
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start = args.reuse_a_layer_start
    head_dim_s = getattr(model_s.config, "head_dim", model_s.config.hidden_size // model_s.config.num_attention_heads)
    head_dim_t = getattr(model_t.config, "head_dim", model_t.config.hidden_size // model_t.config.num_attention_heads)

    mlps = AdapterBank(
        [KAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)],
        [VAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)]
        ).to(device)
    optimizer = optim.Adam(mlps.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    # Data loader
    dataset = HotPotQADataset({"train": args.train_file}, num=1000, split="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    # Prepare with accelerator
    model_s, model_t, mlps, optimizer, dataloader = accelerator.prepare(
        model_s, model_t, mlps, optimizer, dataloader
    )
    
    grad_accum_steps = args.grad_accum_steps
    for epoch in range(args.epochs):
        mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            example = batch[0]
            input_ids = torch.tensor([get_input_ids_list(tokenizer, example)], device=accelerator.device)

            with torch.no_grad():
                s_out = model_s(input_ids=input_ids, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                pkv_s = s_out.past_key_values
                pkv_t = t_out.past_key_values

            loss = torch.tensor(0.0, device=accelerator.device)
            cnt = 0
            for i in range(args.reuse_a_layer_start, N_t):
                s_idx = map_layer_nearest(i, N_s, N_t)
                k_s, v_s = pkv_s[s_idx]
                k_t, v_t = pkv_t[i]
                k_adapted = mlps(k_s, s_idx)
                v_adapted = mlps(v_s, s_idx)
                loss += mse_loss(k_adapted, k_t.to(dtype=k_adapted.dtype)) + mse_loss(v_adapted, v_t.to(dtype=v_adapted.dtype))
                cnt += 2

            if cnt > 0:
                loss = loss / cnt

            loss = loss / grad_accum_steps
            accelerator.backward(loss)
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
            total_loss += avg_loss * grad_accum_steps

            if accelerator.is_main_process:
                pbar.set_postfix({"loss": total_loss / (step + 1)})

        if accelerator.is_main_process and epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")
            save_path = os.path.join(args.output_dir, f"mlp_kv_adapters_epoch{epoch+1}.pth")
            unwrapped_mlps = accelerator.unwrap_model(mlps)
            accelerator.save(unwrapped_mlps.state_dict(), save_path)
            print(f"✅ Saved to {save_path}")

# ---------------- Evaluation ----------------
def evaluate_distr(args):
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).eval()
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).eval()

    # Rebuild MLP and load weights
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start = args.reuse_a_layer_start
    head_dim_s = getattr(model_s.config, "head_dim", ...)
    head_dim_t = getattr(model_t.config, "head_dim", ...)

    mlps = AdapterBank(
        [KAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)],
        [VAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)]
        ).to(device)

    ckpt = os.path.join(args.output_dir, f"mlp_kv_adapters_epoch{args.epochs}.pth")
    state_dict = torch.load(ckpt, map_location="cpu")
    mlps.load_state_dict(state_dict)
    mlps.eval()

    # Prepare
    model_s, model_t, mlps = accelerator.prepare(model_s, model_t, mlps)

    # Load eval data
    ds = load_dataset("parquet", data_files={"valid": args.valid_file})["valid"]
    total_em = total_f1 = 0.0

    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating", disable=not accelerator.is_main_process)
        for step, ex in enumerate(pbar):
            pred, em, f1 = kv_bridged_generate(model_t, model_s, tokenizer, mlps, ex, args)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            if accelerator.is_main_process:
                pbar.set_postfix({"EM": avg_em, "F1": avg_f1})

    if accelerator.is_main_process:
        print(f"Evaluation - EM: {avg_em:.4f}, F1: {avg_f1:.4f}")
   
def l2norm(tensor):
    return torch.sqrt(torch.sum(tensor ** 2))   
        
def train(args):
    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).to(device)
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device)
    model_s.eval().requires_grad_(False)
    model_t.eval().requires_grad_(False)

    # Build MLP adapters
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start = args.reuse_a_layer_start
    head_dim_s = getattr(model_s.config, "head_dim", model_s.config.hidden_size // model_s.config.num_attention_heads)
    head_dim_t = getattr(model_t.config, "head_dim", model_t.config.hidden_size // model_t.config.num_attention_heads)

    k_mlps = AdapterBank(
        KAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)
        ).to(device)
    
    v_mlps = AdapterBank(
        VAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)
        ).to(device)
    
    mse_loss = nn.MSELoss()

    # Data loader
    dataset = HotPotQADataset({"train": args.train_file}, num=1000, split="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    dataset = ["What is the capital of France?", "What is the capital of China?"]

    grad_accum_steps = args.grad_accum_steps

    '''
    print("Starting training k adapter...")
    optimizer = optim.Adam(k_mlps.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        k_mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            #example = batch[0]
            example = batch
            input_ids = torch.tensor([get_input_ids_list(tokenizer, example)], device=device)
            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            with torch.no_grad():
                s_out = model_s(input_ids=input_ids, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                pkv_s = s_out.past_key_values
                pkv_t = t_out.past_key_values

            loss = torch.tensor(0.0, device=device)
            dist = torch.tensor(0.0, device=device)
            cnt = 0
            k_adapted_norm,k_norm = 0.0, 0.0
            for i in range(args.reuse_a_layer_start, N_t):
                s_idx = map_layer_nearest(i, N_s, N_t)
                k_s, v_s = pkv_s[s_idx]
                k_t, v_t = pkv_t[i]
                k_adapted = k_mlps(k_s, s_idx)
                loss += mse_loss(k_adapted, k_t.to(dtype=k_adapted.dtype)) 
                dist += mse_loss(k_s.to(dtype=k_adapted.dtype), k_t.to(dtype=k_adapted.dtype))
                k_adapted_norm += l2norm(k_adapted).item()
                k_norm += l2norm(k_t.to(dtype=k_adapted.dtype)).item()
                cnt += 1

            if cnt > 0:
                loss = loss / cnt

            loss = loss / grad_accum_steps
            loss.backward(loss)
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = loss.item()
            total_loss += avg_loss * grad_accum_steps

            pbar.set_postfix({"loss": total_loss / (step+1),"dist": dist.item()/cnt})
            
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")
        if epoch % 15 ==0 or epoch == args.epochs -1:
            save_path = os.path.join(args.output_dir, f"mlp_k_adapters_epoch{epoch+1}.pth")
            torch.save(k_mlps.state_dict(), save_path)
            print(f"✅ Saved to {save_path}")
    '''
    
            
    print("Starting training v adapter...")
    optimizer = optim.Adam(v_mlps.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        v_mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(pbar):
            example = batch[0]
            input_ids = torch.tensor([get_input_ids_list(tokenizer, example)], device=device)
            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            with torch.no_grad():
                s_out = model_s(input_ids=input_ids, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                pkv_s = s_out.past_key_values
                pkv_t = t_out.past_key_values

            loss = torch.tensor(0.0, device=device)
            dist = torch.tensor(0.0, device=device)
            cnt = 0
            for i in range(args.reuse_a_layer_start, N_t):
                s_idx = map_layer_nearest(i, N_s, N_t)
                k_s, v_s = pkv_s[s_idx]
                k_t, v_t = pkv_t[i]
                v_adapted = v_mlps(v_s, s_idx)
                loss += mse_loss(v_adapted, v_t.to(dtype=v_adapted.dtype)) 
                dist += mse_loss(v_s.to(dtype=v_adapted.dtype), v_t.to(dtype=v_adapted.dtype))
                cnt += 1

            if cnt > 0:
                loss = loss / cnt

            loss = loss / grad_accum_steps
            loss.backward(loss)
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = loss.item()
            total_loss += avg_loss * grad_accum_steps

            pbar.set_postfix({"loss": total_loss / (step + 1),"dist": dist.item()/cnt})

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")
    
        if epoch % 15 ==0 or epoch == args.epochs -1:
            save_path = os.path.join(args.output_dir, f"mlp_v_adapters_epoch{epoch+1}.pth")
            torch.save(v_mlps.state_dict(), save_path)
            print(f"✅ Saved to {save_path}")
            

# ---------------- Evaluation ----------------
def evaluate(args, adpater=False):

    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).to(device).eval()
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device).eval()

    # Rebuild MLP and load weights
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start = args.reuse_a_layer_start
    head_dim_s = getattr(model_s.config, "head_dim", ...)
    head_dim_t = getattr(model_t.config, "head_dim", ...)

    if adpater:
        print("Using provided adapter for evaluation.")
        k_mlps = AdapterBank(
            [KAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t)],
            ).to(device)
        ckpt = os.path.join(args.output_dir, f"mlp_k_adapters_epoch{100}.pth")
        state_dict = torch.load(ckpt, map_location=device)
        k_mlps.load_state_dict(state_dict)
        k_mlps.eval()
        
        v_mlps = AdapterBank(
            [VAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t)],
            ).to(device)
        ckpt = os.path.join(args.output_dir, f"mlp_v_adapters_epoch{100}.pth")
        state_dict = torch.load(ckpt, map_location=device)
        v_mlps.load_state_dict(state_dict)
        v_mlps.eval()
        
    else:
        print("No adapter provided, skipping evaluation.")
        k_mlps, v_mlps = None, None

    # Load eval data
    ds = HotPotQADataset({"valid": args.valid_file}, num=300, split="valid")
    print(f"Evaluating on {args.valid_file} with {len(ds)} examples.")
    
    total_em = total_f1 = 0.0
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            pred, em, f1 = eval_one(ex, tokenizer, model_t, model_s, k_mlps ,v_mlps,args)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
    with open(f"qa_result.log", "a") as f:
        f.write(f"Evaluation {args.reuse_a_layer_start}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")

def eval_wo_reuse(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).to(device).eval()
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device).eval()


    # Load eval data
    ds = HotPotQADataset({"valid": args.valid_file}, num=100, split="valid")
    
    total_em = total_f1 = 0.0
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            input_ids_list = get_input_ids_list(tokenizer, ex)
            pred = pure_generate(model_t,tokenizer, input_ids_list, args)
            gold = ex["answer"]
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
    print(f"Evaluation {args.model_t}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}")

    total_em = total_f1 = 0.0
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            input_ids_list = get_input_ids_list(tokenizer, ex)
            pred = pure_generate(model_s,tokenizer, input_ids_list, args)
            gold = ex["answer"]
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
    print(f"Evaluation {args.model_s}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}")
    
def test(args):
    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).to(device).eval()
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start = args.reuse_a_layer_start
    head_dim_s = getattr(model_s.config, "head_dim", model_s.config.hidden_size // model_s.config.num_attention_heads)
    head_dim_t = getattr(model_t.config, "head_dim", model_t.config.hidden_size // model_t.config.num_attention_heads)  
    
    k_mlps = AdapterBank(
            [KAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)],
            ).to(device)
    ckpt = os.path.join(args.output_dir, f"mlp_k_adapters_epoch{1000}.pth")
    state_dict = torch.load(ckpt, map_location=device)
    k_mlps.load_state_dict(state_dict)
    k_mlps.eval()
        
    v_mlps = AdapterBank(
            [VAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)],
            ).to(device)
    ckpt = os.path.join(args.output_dir, f"mlp_v_adapters_epoch{1000}.pth")
    state_dict = torch.load(ckpt, map_location=device)
    v_mlps.load_state_dict(state_dict)
    v_mlps.eval()
    
    
    ds = HotPotQADataset({"train": args.valid_file}, num=10, split="train")
    ds = ["What is the capital of France?", "What is the capital of China?"]
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            input_ids_list = get_input_ids_list(tokenizer, ex)
            text = tokenizer.decode(input_ids_list, skip_special_tokens=True)
            model_s_out = model_s(input_ids=torch.tensor([input_ids_list], device=device), use_cache=True)
            model_t_out = model_t(input_ids=torch.tensor([input_ids_list], device=device), use_cache=True)
            pkv_s = model_s_out.past_key_values
            pkv_t = model_t_out.past_key_values
            pkv_s = tuple(tuple(t for t in layer) for layer in pkv_s)
            pkv_t = tuple(tuple(t for t in layer) for layer in pkv_t)
            new_cache,_ = reuse_layer_with_mlp(pkv_t,pkv_s,k_mlps,v_mlps,args)
            kloss = torch.tensor(0.0, device=device)
            vloss = torch.tensor(0.0, device=device)
            kloss_a = torch.tensor(0.0, device=device)
            vloss_a = torch.tensor(0.0, device=device)
            vloss_ = torch.tensor(0.0, device=device)
            sk_norm, sv_norm = 0.0, 0.0
            tk_norm, tv_norm = 0.0, 0.0
            ak_norm, av_norm = 0.0, 0.0
            cnt = 0
            for i in range(args.reuse_a_layer_start, N_t):
                s_idx = i
                k_s, v_s = pkv_s[s_idx]
                k_t, v_t = pkv_t[i]
                k_a, v_a = new_cache[i]
                kloss += nn.MSELoss()(k_s, k_t.to(dtype=k_s.dtype)) 
                vloss += nn.MSELoss()(v_s, v_t.to(dtype=v_s.dtype))
                kloss_a += nn.MSELoss()(k_a, k_t.to(dtype=k_a.dtype))
                vloss_a += nn.MSELoss()(v_a, v_t.to(dtype=v_a.dtype))
                vloss_ += nn.MSELoss()(v_a, v_s.to(dtype=v_a.dtype))
                tk_norm += l2norm(k_t.to(dtype=k_s.dtype)).item()
                tv_norm += l2norm(v_t.to(dtype=v_s.dtype)).item()
                sk_norm += l2norm(k_s).item()
                sv_norm += l2norm(v_s).item()
                ak_norm += l2norm(k_a).item()
                av_norm += l2norm(v_a).item()
                cnt += 1
            print(f"kloss:{(kloss/cnt).item()},vloss:{(vloss/cnt).item()}")
            print(f"kloss_a:{(kloss_a/cnt).item()},vloss_a:{(vloss_a/cnt).item()}")
            print(f"vloss_:{(vloss_/cnt).item()}")
            print(f"tk_norm:{(tk_norm/cnt):.4f},tv_norm:{(tv_norm/cnt):.4f}")
            print(f"sk_norm:{(sk_norm/cnt):.4f},sv_norm:{(sv_norm/cnt):.4f}")
            print(f"ak_norm:{(ak_norm/cnt):.4f},av_norm:{(av_norm/cnt):.4f}")

def test1(args):
    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).to(device).eval()
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start = args.reuse_a_layer_start
    head_dim_s = getattr(model_s.config, "head_dim", model_s.config.hidden_size // model_s.config.num_attention_heads)
    head_dim_t = getattr(model_t.config, "head_dim", model_t.config.hidden_size // model_t.config.num_attention_heads)  
    
    k_mlps = AdapterBank(
            [KAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)],
            ).to(device)
    ckpt = os.path.join(args.output_dir, f"mlp_k_adapters_epoch{841}.pth")
    state_dict = torch.load(ckpt, map_location=device)
    k_mlps.load_state_dict(state_dict)
    k_mlps.eval()
        
    v_mlps = AdapterBank(
            [VAdapter(head_dim_s, head_dim_s * 2, head_dim_t) for _ in range(N_t - start)],
            ).to(device)
    ckpt = os.path.join(args.output_dir, f"mlp_v_adapters_epoch{886}.pth")
    state_dict = torch.load(ckpt, map_location=device)
    v_mlps.load_state_dict(state_dict)
    v_mlps.eval()
    
    
    ds = HotPotQADataset({"train": args.valid_file}, num=10, split="train")
    ds = ["What is the capital of France?", "What is the capital of China?"]   
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            input_ids_list = get_input_ids_list(tokenizer, ex)
            text = tokenizer.decode(input_ids_list, skip_special_tokens=True)
            print("text:",text)
            pred = kv_bridged_generate(model_t,model_s,tokenizer,None,None,input_ids_list,args) 
            print("pred:",pred)
    

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_t", type=str, required=True)
    parser.add_argument("--model_s", type=str, required=True)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--reuse_a_layer_start", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--k_reuse", action="store_true")
    parser.add_argument("--v_reuse", action="store_true")
    parser.add_argument("--test1", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test:
        test(args)
    elif args.test1:
        test1(args)
    else:
        if args.train:
            train(args)
        if args.eval:
            # eval_wo_reuse(args)
            evaluate(args, adpater=args.adapter)