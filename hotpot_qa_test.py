import os
os.environ["HF_HOME"] = "/home/gehao/lyz/data/hf-cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse, re, string, math, random
from typing import Dict, List, Tuple, Optional
from collections import Counter
import collections
import json
import math
import os
from tqdm import tqdm
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator
import gc

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM
)
import evaluate
import re
from transformers.cache_utils import DynamicCache

import signal
import sys

def handler(sig, frame):
    print("\n🔴 Ctrl+C detected, terminating all processes...")
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def sample_next_token_batch(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """
    logits: [B, vocab_size] (already the last time step)
    returns: [B] int token id
    """
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).long().to(device)
        
    # Temperature
    logits = logits / temperature

    # Nucleus (top-p)
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return choice.long().to(device)
    else:
        choice = torch.multinomial(probs, num_samples=1)
        return choice.long().to(device)


def reuse_layer(a_cache, b_cache, args):
    """
    a_cache: [28, 2, [b, seqlen, 8, 128]]
    b_cache: [36, 2, [b, seqlen, 8, 128]]

    return: new_a_cache, reuse_b_layer_list
    note that different layers in a_cache and b_cache may be on different devices
    and new_a_cache should keep the same device placement as a_cache
    """
    
    def map_layer_nearest(idx_t, n_layers_s, n_layers_t):
        if n_layers_t <= 1:
            return 0
        return int(round(idx_t * (n_layers_s - 1) / (n_layers_t - 1)))
    
    #print(f"a_cache: {len(a_cache)} layers, b_cache: {len(b_cache)} layers")
    
    reuse_a_layer_start = args.reuse_a_layer_start
    a_kv_cache_list = [(a_cache[layer_idx][0].to(device), a_cache[layer_idx][1].to(device)) for layer_idx in range(reuse_a_layer_start, len(a_cache))]
    b_kv_cache_list = [(b_cache[layer_idx][0].to(device), b_cache[layer_idx][1].to(device)) for layer_idx in range(len(b_cache))]

    reuse_b_layer_list = [map_layer_nearest(layer_idx,len(a_cache),len(b_cache)) for layer_idx in range(reuse_a_layer_start, len(a_cache))]
    reused_a_cache = [b_kv_cache_list[reuse_b_layer_list[i]] for i in range(len(reuse_b_layer_list))]
    new_a_cache = a_cache[:reuse_a_layer_start] + tuple(reused_a_cache)
    return new_a_cache, reuse_b_layer_list


@torch.no_grad()
def kv_bridged_generate(model_t,model_s, tok, mlps, input_ids_list: list[int], args):
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
    new_a_cache, reuse_b_layer_list = reuse_layer_with_mlp(pkv_s, pkv_t, mlps, device, args)
    #print(f"reuse_b_layer_list: {reuse_b_layer_list}")
    
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
    print("text:",text)
    pos = text.index(":")
    if pos>=0:
        text=text[pos+1:].strip()
    return postprocess(text)

@torch.no_grad()
def kv_bridged_generate_batch(model_t,model_s, tok, input_ids_list: list[list[int]], args):
    """return: generated string"""

    eos_id = tok.eos_token_id
    max_new = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p

    input_ids = torch.tensor(input_ids_list, device=device)

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
    new_a_cache, reuse_b_layer_list = reuse_layer(pkv_s, pkv_t, args)
    #print(f"reuse_b_layer_list: {reuse_b_layer_list}")
    
    first_token = sample_next_token_batch(s_out.logits[:,-1,:], temperature, top_p)
    
    past = DynamicCache.from_legacy_cache(past_key_values=new_a_cache)
    generated = [first_token.squeeze(-1).tolist()]  # [B]
    last_token = first_token  # [B, 1]
    print(last_token.shape)
    for _ in range(max_new-1):
        out = model_t(
            input_ids=last_token,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        logits = out.logits[:, -1, :]  # [vocab]
        past = out.past_key_values              # now past is from A going forward

        next_id = sample_next_token_batch(logits, temperature, top_p)
        generated.append(next_id.squeeze(-1).tolist())  # [B]

        if all([i==eos_id for i in generated[-1]]):
            break

        # Prepare inputs for next step
        last_token = next_id
    
    # generated: [seq_len, B] -> [B, seq_len]
    generated = list(map(list, zip(*generated)))
    print("generated:",torch.tensor(generated).shape)
    # decode each generated[i]
    text = []
    for i in range(len(generated)):
        generated[i] = [t for t in generated[i] if t != eos_id]
        text.append(tok.decode(generated[i], skip_special_tokens=True))
    print("text:",text)
    return [postprocess(t) for t in text]


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

def eval_one(example: Dict, tok, model_t, model_s,mlps, args) -> Tuple[str, float, float, str]:
    input_ids_list = get_input_ids_list(tok, example)
    pred = kv_bridged_generate(model_t,model_s, tok, mlps, input_ids_list, args)
    gold = example["answer"]
    with open(f"debug2.log", "a") as f:
        f.write(f"Q: {example['question']}\nA: {pred}\nG: {gold}\n")
    return pred, exact_match(pred, gold), f1_score(pred, gold)

# ---------------- MLP Adapter ----------------
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
            nn.LayerNorm(output_size)  # Stabilize outputs
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
def reuse_layer_with_mlp(a_cache, b_cache, mlps: nn.ModuleList, device, args):
    """
    a_cache: [N_a, 2, (B, H, S, D)]  ← student KV cache
    b_cache: [N_b, 2, (B, H, S, D)]  ← teacher KV cache
    mlps: ModuleList of K MLPs, one for each reused layer
    Transforms selected b_cache KV states using MLP and injects into a_cache.
    """
    def map_layer_nearest(idx_t, n_layers_s, n_layers_t):
        if n_layers_t <= 1:
            return 0
        return int(round(idx_t * (n_layers_s - 1) / (n_layers_t - 1)))

    reuse_a_layer_start = args.reuse_a_layer_start

    # Map student layers to teacher layers
    reuse_b_layer_list = [
        map_layer_nearest(layer_idx, len(a_cache), len(b_cache))
        for layer_idx in range(reuse_a_layer_start, len(a_cache))
    ]

    adapted_kv_list = []
    for i, b_idx in enumerate(reuse_b_layer_list):
        key_t, val_t = b_cache[b_idx]  # Teacher KV at b_idx
        # Apply MLP adapter
        k=key_t.to(device)
        v=val_t.to(device)
        key_adapted = mlps(k, i)
        val_adapted = mlps(v, i)
        adapted_kv_list.append((key_adapted, val_adapted))
        #adapted_kv_list.append((k,v))

    # Replace from `reuse_a_layer_start` onward
    new_a_cache = list(a_cache[:reuse_a_layer_start]) + adapted_kv_list
    return tuple(new_a_cache), reuse_b_layer_list

# ---------------- Training Loop ----------------
def train_kv_adapter(
    model_t,
    model_s,
    tokenizer,
    train_examples,
    args
):
    # Freeze teacher and student models
    for param in model_t.parameters():
        param.requires_grad = False
    for param in model_s.parameters():
        param.requires_grad = False

    # Setup: assume student has N_t layers, and we start reusing from `reuse_a_layer_start`
    N_s = len(model_s.model.layers)  # Adjust based on actual architecture
    N_t = len(model_t.model.layers)
    start_layer = args.reuse_a_layer_start
    num_reused = N_t - start_layer

    # Initialize MLP adapters
    head_dim_t = model_t.config.head_dim
    kv_dim_t = head_dim_t  # Assuming standard multi-head attention
    head_dim_s = model_s.config.head_dim
    kv_dim_s = head_dim_s  # Assuming standard multi-head attention

    mlps = AdapterBank([
        KVAdapter(input_size=kv_dim_s, hidden_size=kv_dim_s * 2, output_size=kv_dim_t)
        for _ in range(num_reused)
    ]).to(device)

    optimizer = optim.Adam(mlps.parameters(), lr=args.lr)
    criterion = lambda pred, gold: 1.0 - f1_score(pred, gold)  # Minimize error

    # Training loop
    mlps.train()
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        print("traning examples:",len(train_examples))
        pbar = tqdm(train_examples, desc=f"Epoch {epoch+1}")
        for example in pbar:
            optimizer.zero_grad()
            # Encode input
            input_ids_list = get_input_ids_list(tokenizer, example)
            input_ids = torch.tensor([input_ids_list], device=args.device)

            # Prefill both models
            with torch.no_grad():
                s_out = model_s(input_ids=input_ids, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                pkv_s = tuple(tuple(t for t in layer) for layer in s_out.past_key_values)
                pkv_t = tuple(tuple(t for t in layer) for layer in t_out.past_key_values)

            # Apply MLP-adapted KV reuse
            new_pkv_t, _ = reuse_layer_with_mlp(pkv_t, pkv_s, mlps, device,args)
            
      
            # First token sampling from student post-transformation
            first_token_logits = s_out.logits[:, -1, :]
            first_token = sample_next_token(first_token_logits.squeeze(0), args.temperature, args.top_p)

            # Decode rest using teacher with adapted KV
            generated = [first_token]
            last_token = torch.tensor([[first_token]], device=args.device)
            past = DynamicCache.from_legacy_cache(past_key_values=new_pkv_t)

            for _ in range(args.max_new_tokens - 1):
                out = model_t(
                    input_ids=last_token,
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                logits = out.logits[:, -1, :].squeeze(0)
                next_token = sample_next_token(logits, args.temperature, args.top_p)
                generated.append(next_token)
                if next_token == tokenizer.eos_token_id:
                    break
                last_token = torch.tensor([[next_token]], device=args.device)
                past = out.past_key_values

            pred_text = tokenizer.decode([t for t in generated if t != tokenizer.eos_token_id], skip_special_tokens=True)
            pred_text = postprocess(pred_text)
            gold_text = example["answer"]

            # Compute loss
            loss = criterion(pred_text, gold_text)
            loss_tensor = torch.tensor(loss, requires_grad=True).to(args.device)
            loss_tensor.backward()
            optimizer.step()

            total_loss += loss
            correct += int(normalize(pred_text) == normalize(gold_text))
            pbar.set_postfix({"loss": loss, "acc": correct / (pbar.n + 1)})
            

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(pbar):.4f}, Acc: {correct/len(pbar):.4f}")

    return mlps
    
def train_kv_adapter_c2c(model_t, model_s, tokenizer, train_examples, args):
    # Freeze models
    model_t.eval()
    model_s.eval()
    
    head_dim_t = model_t.config.head_dim
    head_dim_s = model_s.config.head_dim
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start, num_reused = args.reuse_a_layer_start, N_s - args.reuse_a_layer_start
    
    mlps = AdapterBank([
        KVAdapter(input_size=head_dim_s, hidden_size=head_dim_s, output_size=head_dim_t)
        for _ in range(num_reused)
    ]).to(device)

    load_path= "mlp_kv_adapters_c2c.pth"
    if os.path.exists(load_path):
        print(f"Loading adapter from {load_path}")
        mlps.load_state_dict(torch.load("mlp_kv_adapters_c2c.pth", map_location=device))
    else:
        print("No existing adapter found, training from scratch.")
    
    
    mlps.train()
    optimizer = optim.Adam(mlps.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(train_examples, desc=f"Epoch {epoch+1}")

        for example in pbar:
            optimizer.zero_grad()

            input_ids_list = get_input_ids_list(tokenizer, example)
            input_ids = torch.tensor([input_ids_list], device=args.device)
            
            with torch.no_grad():
                # Prefill both models
                s_out = model_s(input_ids=input_ids, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)

                pkv_s = [(k.detach(), v.detach()) for k, v in s_out.past_key_values]
                pkv_t = [(k.detach(), v.detach()) for k, v in t_out.past_key_values]

            loss = 0.0
            cnt = 0
            for i in range(args.reuse_a_layer_start, len(model_t.model.layers)):
                layer_s_idx = i
                k_s, v_s = pkv_s[layer_s_idx]  # [B, nh, seq_len, d_k]
                k_t, v_t = pkv_t[i]

                k_adapted = mlps(k_s,i - args.reuse_a_layer_start)
                v_adapted = mlps(v_s,i - args.reuse_a_layer_start)

                loss += mse_loss(k_adapted, k_t.to(k_adapted.dtype)) 
                loss += mse_loss(v_adapted, v_t.to(v_adapted.dtype))
                cnt += 2

            if cnt > 0:
                loss = loss / cnt

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

            # Optional cleanup
            del loss; torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(pbar):.4f}")
        
    torch.save(mlps.state_dict(), "mlp_kv_adapters_c2c.pth")
    
    return mlps

def eval(model_t, model_s, tokenizer, eval_examples: List[Dict], args) -> Tuple[float, float]:
    
    mlps = AdapterBank([
        KVAdapter(input_size=model_s.config.head_dim, hidden_size=model_s.config.head_dim, 
                  output_size=model_t.config.head_dim)
        for _ in range(len(model_t.model.layers) - args.reuse_a_layer_start)
    ]).to(device)
    
    mlp_path="adapter_out/mlps_epoch1.pt"
    if os.path.exists(mlp_path):
        print(f"Loading MLP adapter from {mlp_path}")
        state_dict = torch.load(mlp_path, map_location=device)
        mlps.load_state_dict(state_dict)
        print("✅ MLP adapter loaded successfully.")
    else:
        raise FileNotFoundError(f"MLP checkpoint not found at {args.mlp_ckpt_path}")
    
    model_t.eval()
    model_s.eval()
    total_em = 0.0
    total_f1 = 0.0
    
    with torch.no_grad():
        avg_em=0.0
        avg_f1=0.0
        pbar = tqdm(eval_examples, desc="Evaluating")
        for step,example in enumerate(pbar):
            pred, em, f1 = eval_one(example, tokenizer, model_t, model_s, mlps,args)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
    print(f"Eval Results - EM: {avg_em:.4f}, F1: {avg_f1:.4f}")
    return avg_em, avg_f1

def train_kv_adapter_c2c_accelerate(tok, args):
    
    def create_dataloader(dataset, batch_size, shuffle=False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.local_process_index,
            shuffle=shuffle
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda x:x ,
            num_workers=1,
            pin_memory=True
        )
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,  # "no", "fp16", "bf16"
        log_with=None
    )
    device = accelerator.device
    print(f"rank: {accelerator.process_index}, Accelerator device:", device)

    # Freeze teacher and student models
    model_s = AutoModelForCausalLM.from_pretrained(args.model_s)
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t)
    
    model_t.eval()
    model_s.eval()

    # Setup dimensions
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start_layer = args.reuse_a_layer_start
    num_reused = N_t - start_layer

    head_dim_s = model_s.config.head_dim
    head_dim_t = model_t.config.head_dim

    # Initialize MLP adapters
    mlps = AdapterBank([
        KVAdapter(input_size=head_dim_s, hidden_size=head_dim_s * 2, output_size=head_dim_t)
        for _ in range(num_reused)
    ])

    # Optimizer
    optimizer = torch.optim.Adam(mlps.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    # Dataset and DataLoader
    train_data = load_dataset("parquet", data_files=args.train_file)
    train_data = train_data["train"]
    train_data=train_data.with_format("torch")

    dataloader = create_dataloader(train_data, batch_size=1, shuffle=True)

    # Prepare with accelerator
    model_t, model_s, mlps, optimizer, dataloader = accelerator.prepare(
        model_t, model_s, mlps, optimizer, dataloader
    )

    # Training loop
    for epoch in range(args.epochs):
        mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)
        #pbar = dataloader
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            with torch.no_grad():
                input_ids_list = get_input_ids_list(tok, batch[0])
                input_ids = torch.tensor([input_ids_list], device=device)
                s_out = model_s(input_ids=input_ids, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)

                pkv_s = [(k.detach(), v.detach()) for k, v in s_out.past_key_values]
                pkv_t = [(k.detach(), v.detach()) for k, v in t_out.past_key_values]

            loss = torch.tensor(0.0, device=accelerator.device)
            cnt = 0

            for i in range(start_layer, N_t):
                layer_s_idx = i
                k_s, v_s = pkv_s[layer_s_idx]
                k_t, v_t = pkv_t[i]

                # Apply MLP adapter
                k_adapted = mlps(k_s,i - start_layer)
                v_adapted = mlps(v_s, i - start_layer)

                loss_k = mse_loss(k_adapted, k_t.to(k_adapted.dtype))
                loss_v = mse_loss(v_adapted, v_t.to(v_adapted.dtype))

                loss = loss + loss_k + loss_v
                cnt += 2

            if cnt > 0:
                loss = loss / cnt

            # Backward pass with accelerate
            accelerator.backward(loss)

            optimizer.step()

            # Gather loss across all processes
            avg_loss = accelerator.gather(loss).mean().item()
            total_loss += avg_loss


            if accelerator.is_main_process:
                pbar.set_postfix({"loss": total_loss / (step + 1)})
            
            
            if step % 50 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Step {step}, Avg Loss: {total_loss / (step + 1):.4f}")
                
            del s_out, t_out
            torch.cuda.empty_cache()

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(dataloader):.4f}")

            # Save only on main process
            
            save_path = os.path.join(args.output_dir, f"mlp_kv_adapters_accel_epoch{epoch+1}.pth")
            
            if  epoch == args.epochs - 1:
                unwrapped_mlps = accelerator.unwrap_model(mlps)
                state_dict = unwrapped_mlps.state_dict()
                accelerator.save(state_dict, save_path)
                print(f"✅ MLP adapters saved to {save_path}")

    return mlps

def layer_map(idx, n_layers_t, n_layers_s):
    if n_layers_t <= 1:
        return 0
    return int(round(idx * (n_layers_s - 1) / (n_layers_t - 1)))

def train_kv_adapter_sft_accelerate(args):
    
    def create_dataloader(dataset, batch_size, shuffle=True):
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.local_process_index,
            shuffle=shuffle
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda x: x,
            num_workers=1,
            pin_memory=True
        )
        
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=None
    )
    print(f"Rank {accelerator.process_index}: Starting SFT training on device {accelerator.device}")

    # 加载模型和 tokenizer
    model_s = AutoModelForCausalLM.from_pretrained(args.model_s)
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t)
    tok = AutoTokenizer.from_pretrained(args.model_t)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model_t.eval()
    model_s.eval()

    # 设置维度
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    start_layer = args.reuse_a_layer_start
    num_reused = N_t - start_layer
    head_dim_s = getattr(model_s.config, "head_dim", model_s.config.hidden_size // model_s.config.num_attention_heads)
    head_dim_t = getattr(model_t.config, "head_dim", model_t.config.hidden_size // model_t.config.num_attention_heads)

    # 初始化 MLP adapters
    mlps = AdapterBank([
        KVAdapter(input_size=head_dim_s, hidden_size=head_dim_s * 2, output_size=head_dim_t)
        for _ in range(num_reused)
    ])

    optimizer = torch.optim.Adam(mlps.parameters(), lr=args.lr)

    # 加载数据集
    ds = load_dataset("parquet", data_files=args.train_file)["train"]
    ds = ds.with_format("torch")
    dataloader = create_dataloader(ds, batch_size=1, shuffle=True)

    # Prepare with accelerator
    mlps, optimizer, dataloader = accelerator.prepare(
        mlps, optimizer, dataloader
    )

    # Training loop
    for epoch in range(args.epochs):
        mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            optimizer.zero_grad()

            example = batch[0]  # 解包
            input_ids_list = get_input_ids_list(tok, example)
            input_ids = torch.tensor([input_ids_list], device=accelerator.device)

            with torch.no_grad():
                # Prefill with student model
                s_out = model_s(input_ids=input_ids, use_cache=True)
                pkv_s = s_out.past_key_values  # List[(k_s, v_s)]
                pkv_s = [(k.detach(), v.detach()) for k, v in pkv_s]

                # Extract and transform KV cache using mlps
                transformed_pkv = []
                
                for layer_idx in range(N_t):
                    if layer_idx < start_layer:
                        continue  # Skip layers before reuse point
                    k_s, v_s = pkv_s[layer_map(layer_idx), N_t, N_s]
                    k_adapted = mlps(k_s, layer_idx)
                    v_adapted = mlps(v_s, layer_idx)
                    transformed_pkv.append((k_adapted, v_adapted))

                # Combine unchanged and transformed caches
                full_pkv = list(pkv_s[:start_layer]) + transformed_pkv

            # Now generate with teacher model using transformed cache
            answer_ids_list = get_answer_ids_list(tok, example)
            label_ids_list  = [-100] * len(input_ids_list) + answer_ids_list
            label_ids = torch.tensor([label_ids_list], device=accelerator.device)
            
            try:
                out = model_t(
                    input_ids=input_ids,
                    past_key_values=full_pkv,
                    labels=label_ids,  # Compute loss
                    use_cache=True
                )
                lm_loss = out.loss  # CrossEntropy over entire sequence
            except Exception as e:
                print(f"Error during generation: {e}")
                continue

            # Backward pass through mlps
            accelerator.backward(lm_loss)
            optimizer.step()

            # Gather and log
            avg_loss = accelerator.gather(lm_loss).mean().item()
            total_loss += avg_loss

            if accelerator.is_main_process:
                pbar.set_postfix({"lm_loss": total_loss / (step + 1)})

            # Clean up
            del s_out, out, full_pkv
            torch.cuda.empty_cache()

        if accelerator.is_main_process:
            print(f"SFT Epoch {epoch+1}, Avg LM Loss: {total_loss / len(pbar):.4f}")
            save_path = os.path.join(args.output_dir, f"mlp_kv_adapters_sft_epoch{epoch+1}.pth")
            unwrapped_mlps = accelerator.unwrap_model(mlps)
            accelerator.save(unwrapped_mlps.state_dict(), save_path)
            print(f"✅ SFT-trained MLP adapters saved to {save_path}")

    return mlps
    

def train(args):
    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device
    print("Accelerator device:", device)

    mse_loss = nn.MSELoss()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_s)
    tokenizer.pad_token = tokenizer.eos_token  # safer than setting pad_token_id directly

    # load frozen models
    model_t = AutoModelForCausalLM.from_pretrained(
        args.model_t,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if args.fp16 else None
    )
    model_s = AutoModelForCausalLM.from_pretrained(
        args.model_s,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if args.fp16 else None
    )

    model_t.eval(); model_s.eval()
    for p in model_t.parameters(): p.requires_grad = False
    for p in model_s.parameters(): p.requires_grad = False
    model_t.to(device)
    model_s.to(device)

    # load dataset

    ds = load_dataset("parquet", data_files={"train": args.train_file})
    train_ds = ds["train"]

    # initialize adapters
    N_s, N_t = len(model_s.model.layers), len(model_t.model.layers)
    start, num_reused = args.reuse_a_layer_start, N_s - args.reuse_a_layer_start
    head_dim_t, head_dim_s = model_t.config.head_dim, model_s.config.head_dim

    mlps = AdapterBank([
        KVAdapter(input_size=head_dim_s, hidden_size=head_dim_s, output_size=head_dim_t)
        for _ in range(num_reused)
    ]).to(device)
    optimizer = optim.Adam(mlps.parameters(), lr=args.lr)

    # distributed sampler ensures each rank gets different data
    sampler = DistributedSampler(train_ds) if accelerator.num_processes > 1 else None

    loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda batch: batch
    )

    # prepare everything with accelerator
    mlps, optimizer, loader = accelerator.prepare(mlps, optimizer, loader)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        progress = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        running_loss = 0.0

        for b,batch in enumerate(progress):
            example = batch[0]
            input_ids_list = get_input_ids_list(tokenizer, example)

            input_ids = torch.tensor([input_ids_list], device=device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            with torch.no_grad():
                s_out = model_s(input_ids=input_ids, use_cache=True, attention_mask=attention_mask)
                t_out = model_t(input_ids=input_ids, use_cache=True, attention_mask=attention_mask)
            
            print(f"model_s: rank {accelerator.process_index}, device: {device}", model_s)
            print(f"s_out logits: rank {accelerator.process_index}, device: {device}", s_out)

            pkv_s = tuple(tuple(t for t in layer) for layer in s_out.past_key_values)
            pkv_t = tuple(tuple(t for t in layer) for layer in t_out.past_key_values)

            
            print("Before reuse_layer_with_mlp, gpu tensors:")
            print(f"skv: rank {accelerator.process_index}, device: {device}", pkv_s)
            print(f"tkv: rank {accelerator.process_index}, device: {device}", pkv_t)
            
            adapted_kv_list, reuse_map = reuse_layer_with_mlp(pkv_t, pkv_s, mlps, device, args)
            print("After reuse_layer_with_mlp, gpu tensors:")
            print(adapted_kv_list)
            # compute loss safely
            total_loss = torch.tensor(0.0, device=device)
            cnt = 0
            for (k_ad, v_ad), (k_s, v_s) in zip(adapted_kv_list, [pkv_s[idx] for idx in reuse_map]):
                k_s_t = k_s.to(device=device, dtype=k_ad.dtype)
                v_s_t = v_s.to(device=device, dtype=v_ad.dtype)
                total_loss = total_loss + mse_loss(k_ad, k_s_t) + mse_loss(v_ad, v_s_t)
                if(accelerator.process_index==1 and b==0):
                    print(k_ad,k_s_t)
                cnt += 2
            if cnt > 0:
                total_loss = total_loss / cnt
            
            print(f"rank: {accelerator.process_index} ","total_loss:",total_loss)

            # backward
            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += total_loss.item()
            if accelerator.is_main_process:
                progress.set_postfix({"loss": running_loss / (progress.n + 1)})

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} finished, avg loss: {running_loss / len(loader):.6f}")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(mlps.state_dict(), f"{args.output_dir}/mlps_epoch{epoch+1}.pt")

    accelerator.free_memory()
    return mlps



if __name__=="__main__":
   
    
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str, default="no", help="no, fp16, bf16")
    parser.add_argument("--train", action="store_true", help="whether to run training")
    parser.add_argument("--eval", action="store_true", help="whether to run eval on validation set")
    args = parser.parse_args()
    

    '''
    model_t=AutoModelForCausalLM.from_pretrained(args.model_t).to(device)
    model_s=AutoModelForCausalLM.from_pretrained(args.model_s).to(device)
    tok=AutoTokenizer.from_pretrained(args.model_s)
    tok.pad_token = tok.eos_token
    print("Models and tokenizer loaded.")
    '''
    
    args_=argparse.Namespace(
        dataset_config="distractor",
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        device=device,
        reuse_a_layer_start=0,  # for Qwen3-1.7B and Qwen3-0.6B
        lr=1e-4,
        epochs=3,
        max_train_steps=100,
    )
    
    data_files = {
        "training": "/home/gehao/lyz/train-00000-of-00002.parquet",
        "validation": "/home/gehao/lyz/validation-00000-of-00001.parquet"
    }

    tok = AutoTokenizer.from_pretrained(args.model_t)
    tok.pad_token = tok.eos_token

    
    if args.train:
        ds = load_dataset("parquet", data_files={"train": args.train_file})
        train_ds = ds["train"]
        mlps=train_kv_adapter_c2c_accelerate(tok,args)
        print("Training completed.")
    
    

    
    else:
        model_s = AutoModelForCausalLM.from_pretrained(args.model_s).to(device)
        model_t = AutoModelForCausalLM.from_pretrained(args.model_t).to(device)
        ds = load_dataset("parquet", data_files={"valid": args.valid_file})
        val_data = ds["valid"]
        em,f1=eval(
            model_t=model_t,
            model_s=model_s,
            tokenizer=tok,
            eval_examples=val_data,
            args=args_
        )

    
    