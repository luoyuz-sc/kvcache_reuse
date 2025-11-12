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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
    text = tok.decode(input_ids_list)
    #print("text:",text)
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
    
    first_token = sample_next_token(t_out.logits[:,-1,:].squeeze(0), temperature, top_p)
    word = tok.decode([first_token])
    #print("word:",word)
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
    with open("debug.log",'a') as f:
        f.write(f"text:{text}\n")
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

    # prefill with s
    t_out = model(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    )
    
    pkv_t = tuple(tuple(t for t in layer) for layer in t_out.past_key_values)
    
    past = DynamicCache.from_legacy_cache(past_key_values=pkv_t)
    
    first_token = sample_next_token(t_out.logits[:,-1,:].squeeze(0), temperature, top_p)
    
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

def format_context_wiki(example: Dict) -> str:
    """Compact, readable multi-hop context."""
    return example["context"]
    
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

def get_input_ids_list_wiki(tokenizer, example: Dict) -> list[int]:
    ctx = format_context_wiki(example)
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

def eval_one(example: Dict, tok, model_t, model_s, k_mlps,v_mlps, args) -> Tuple[str, float, float]:
    input_ids_list = get_input_ids_list(tok, example)
    pred = kv_bridged_generate(model_t,model_s, tok, k_mlps,v_mlps, input_ids_list, args)
    gold = example["answer"]
    with open(f"./log/qa_result.log", "a") as f:
        f.write(f"Q: {example['question']}\nA: {pred}\nG: {gold}\n")
    return pred, exact_match(pred, gold), f1_score(pred, gold)

def eval_one_wiki(example: Dict, tok, model_t, model_s, k_mlps,v_mlps, args) -> Tuple[str, float, float]:
    input_ids_list = get_input_ids_list_wiki(tok, example)
    pred = kv_bridged_generate(model_t,model_s, tok, k_mlps,v_mlps, input_ids_list, args)
    gold = example["answer"]
    with open(f"./log/wikiqa_result.log", "a") as f:
        f.write(f"Q: {example['question']}\nA: {pred}\nG: {gold}\n")
    return pred, exact_match(pred, gold), f1_score(pred, gold)

def evalone_wo_reuse(example: Dict, tok, model, args) -> Tuple[str, float, float]:
    input_ids_list = get_input_ids_list(tok, example)
    pred = pure_generate(model, tok,input_ids_list, args)
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

class RegularMLP(nn.Module):
    """
    Regular MLP used as a drop-in adapter block.
    API expected by earlier code: RegularMLP(hidden_dim, intermediate_dim, num_layers)
    Input:  [N, hidden_dim]
    Output: [N, hidden_dim]

    Implementation details:
    - num_layers blocks. Each block: Linear(hidden_dim -> intermediate_dim) -> Activation -> Dropout -> Linear(intermediate_dim -> hidden_dim)
    - Residual connection around each block (so block input and output both have hidden_dim)
    - Optional LayerNorm after each residual add (default: off)
    - Default activation: GELU
    """
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_layers: int = 3,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        assert hidden_dim > 0 and intermediate_dim > 0 and num_layers >= 1
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.use_layernorm = use_layernorm

        self.blocks = nn.ModuleList()
        if use_layernorm:
            self.norms = nn.ModuleList()

        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
            )
            self.blocks.append(block)
            if use_layernorm:
                self.norms.append(nn.LayerNorm(hidden_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize Linear layers with xavier uniform (common choice for MLPs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, hidden_dim]
        returns: [N, hidden_dim]
        """
        assert x.dim() == 2 and x.size(1) == self.hidden_dim, f"Expected input shape [N, {self.hidden_dim}]"
        out = x
        for i, block in enumerate(self.blocks):
            res = block(out)         # [N, hidden_dim]
            out = out + res          # residual
            if self.use_layernorm:
                out = self.norms[i](out)
        return out

class KAdapter_(nn.Module):
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
    
class VAdapter_(nn.Module):
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
class KAdapter(nn.Module):
    """
    Adapter that converts teacher K (multiple heads) -> student-expected K shape.
    Input:  x shape [B, send_heads, S, send_head_dim]
    Output: key_out shape [B, receive_heads , S, receive_head_dim]
    """
    def __init__(
        self,
        send_heads: int,
        send_head_dim: int,
        receive_heads: int,
        receive_head_dim: int,
        hidden_dim: int = None,
        intermediate_dim: int = None,
        num_layers: int = 3,
    ):
        super().__init__()
        self.send_heads = send_heads
        self.send_head_dim = send_head_dim
        self.receive_heads = receive_heads
        self.receive_head_dim = receive_head_dim

        # default hidden/intermediate sizes if not provided
        if hidden_dim is None:
            hidden_dim = send_heads * send_head_dim  # simple default
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 2

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers

        # embed：把多个 send heads 拼成一个向量后投到 hidden 空间
        self.key_embed = nn.Linear(send_heads * send_head_dim, hidden_dim)

        # MLPs（仿照 Translator 的结构）
        self.key_mlp = RegularMLP(hidden_dim, intermediate_dim, num_layers)

        # 输出投回 receive_heads * receive_head_dim，然后 reshape 成多头格式
        self.key_out = nn.Linear(hidden_dim, receive_heads * receive_head_dim)

        # 对输出向量做归一化（和你原来做法一致，把 norm 放在输出维度上）
        self.key_norm = RMSNorm(receive_heads * receive_head_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [B, send_heads, S, send_head_dim]
        returns: (k_out, v_out) each [B, receive_heads , S, receive_head_dim]
        """
        assert x.dim() == 4, "expected input shape [B, S, send_heads, send_head_dim]"
        B, Sh, S, D = x.shape
        assert Sh == self.send_heads and D == self.send_head_dim, (
            f"input heads/dim mismatch: got ({Sh},{D}), expected "
            f"({self.send_heads},{self.send_head_dim})"
        )

        # 1) 把 heads 拼在最后一个维度： [B, S, send_heads * send_head_dim]
        x_comb = x.permute(0,2,1,3).reshape(B, S, Sh * D)

        # 2) key path
        k = self.key_embed(x_comb)            # [B, S, hidden_dim]
        k_flat = k.reshape(B * S, self.hidden_dim)
        k_hidden = self.key_mlp(k_flat)       # RegularMLP 期望 [N, hidden_dim] -> [N, hidden_dim]
        k_out_flat = self.key_out(k_hidden)   # [B*S, receive_heads * receive_head_dim]
        k_out_flat = self.key_norm(k_out_flat)  # RMSNorm 在最后一维归一化
        k_out = k_out_flat.view(B, S, self.receive_heads, self.receive_head_dim)
        k_out = k_out.permute(0,2,1,3)

        return k_out
    
class KVAdapter(nn.Module):
    """
    Adapter that converts teacher K/V (multiple heads) -> student-expected K/V shape.

    Input:
        k: [B, send_heads, S, send_head_dim]
        v: [B, send_heads, S, send_head_dim]
    Output:
        k_out: [B, receive_heads, S, receive_head_dim]
        v_out: [B, receive_heads, S, receive_head_dim]
    """
    def __init__(
        self,
        send_heads: int,
        send_head_dim: int,
        receive_heads: int,
        receive_head_dim: int,
        hidden_dim: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
        num_layers: int = 2,
        share_mlp: bool = False,
    ):
        super().__init__()
        self.send_heads = send_heads
        self.send_head_dim = send_head_dim
        self.receive_heads = receive_heads
        self.receive_head_dim = receive_head_dim

        if hidden_dim is None:
            hidden_dim = send_heads * send_head_dim
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 2

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.share_mlp = share_mlp

        # Embedding layers (project concatenated heads -> hidden)
        # If share_mlp True we use the same embed for key and value
        self.key_embed = nn.Linear(send_heads * send_head_dim, hidden_dim)
        if not share_mlp:
            self.value_embed = nn.Linear(send_heads * send_head_dim, hidden_dim)
        else:
            self.value_embed = None

        # MLPs
        self.key_mlp = RegularMLP(hidden_dim, intermediate_dim, num_layers)
        if not share_mlp:
            self.value_mlp = RegularMLP(hidden_dim, intermediate_dim, num_layers)
        else:
            self.value_mlp = self.key_mlp  # share weights

        # Output projections: hidden -> receive_heads * receive_head_dim
        self.key_out = nn.Linear(hidden_dim, receive_heads * receive_head_dim)
        self.value_out = nn.Linear(hidden_dim, receive_heads * receive_head_dim)

        # Norms on final flattened vectors
        self.key_norm = RMSNorm(receive_heads * receive_head_dim)
        self.value_norm = RMSNorm(receive_heads * receive_head_dim)

    def forward(self, k: torch.Tensor, v: torch.Tensor, layer_idx: Optional[int] = None):
        """
        k, v: [B, send_heads, S, send_head_dim]
        returns: k_out, v_out each [B, receive_heads, S, receive_head_dim]
        layer_idx: optional, ignored by default (kept for compatibility)
        """
        assert k.dim() == 4 and v.dim() == 4, "expected input shape [B, send_heads, S, send_head_dim]"
        Bk, Sh_k, S_k, Dk = k.shape
        Bv, Sh_v, S_v, Dv = v.shape
        assert Bk == Bv and Sh_k == self.send_heads and Sh_v == self.send_heads, "batch/heads mismatch"
        assert Dk == self.send_head_dim and Dv == self.send_head_dim, "head_dim mismatch"
        assert S_k == S_v, "sequence length mismatch between k and v"
        B, S = Bk, S_k
        Sh, D = Sh_k, Dk

        # combine heads dimension -> [B, S, send_heads * send_head_dim]
        # input layout is [B, send_heads, S, send_head_dim], so permute then reshape
        k_comb = k.permute(0, 2, 1, 3).reshape(B, S, Sh * D)  # [B, S, Sh*D]
        v_comb = v.permute(0, 2, 1, 3).reshape(B, S, Sh * D)

        # key path
        k_hidden = self.key_embed(k_comb)          # [B, S, hidden_dim]
        k_flat = k_hidden.reshape(B * S, self.hidden_dim)
        k_processed = self.key_mlp(k_flat)         # [B*S, hidden_dim]
        k_out_flat = self.key_out(k_processed)     # [B*S, receive_heads * receive_head_dim]
        k_out_flat = self.key_norm(k_out_flat)
        k_out = k_out_flat.view(B, S, self.receive_heads, self.receive_head_dim).permute(0, 2, 1, 3)

        # value path (may share MLP)
        if self.value_embed is not None:
            v_hidden = self.value_embed(v_comb)    # separate embed
            v_flat = v_hidden.reshape(B * S, self.hidden_dim)
            v_processed = self.value_mlp(v_flat)
        else:
            # share embed & mlp: reuse key embed/mlp but feed v_comb
            v_hidden = self.key_embed(v_comb)
            v_flat = v_hidden.reshape(B * S, self.hidden_dim)
            v_processed = self.key_mlp(v_flat)    # same weights as key_mlp when share_mlp=True

        v_out_flat = self.value_out(v_processed)
        v_out_flat = self.value_norm(v_out_flat)
        v_out = v_out_flat.view(B, S, self.receive_heads, self.receive_head_dim).permute(0, 2, 1, 3)

        return k_out, v_out


class VAdapter(nn.Module):
    """
    Adapter that converts teacher V (multiple heads) -> student-expected V shape.
    Input:  x shape [B, send_heads, S, send_head_dim]
    Output: key_out shape [B, receive_heads , S, receive_head_dim]
    """
    def __init__(
        self,
        send_heads: int,
        send_head_dim: int,
        receive_heads: int,
        receive_head_dim: int,
        hidden_dim: int = None,
        intermediate_dim: int = None,
        num_layers: int = 2,
    ):
        super().__init__()
        self.send_heads = send_heads
        self.send_head_dim = send_head_dim
        self.receive_heads = receive_heads
        self.receive_head_dim = receive_head_dim

        # default hidden/intermediate sizes if not provided
        if hidden_dim is None:
            hidden_dim = send_heads * send_head_dim  # simple default
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 2

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers

        # embed：把多个 send heads 拼成一个向量后投到 hidden 空间
        self.v_embed = nn.Linear(send_heads * send_head_dim, hidden_dim)

        # MLPs（仿照 Translator 的结构）
        self.v_mlp = RegularMLP(hidden_dim, intermediate_dim, num_layers)

        # 输出投回 receive_heads * receive_head_dim，然后 reshape 成多头格式
        self.v_out = nn.Linear(hidden_dim, receive_heads * receive_head_dim)


    def forward(self, x: torch.Tensor):
        """
        x: [B, send_heads, S, send_head_dim]
        returns: v_out [B, receive_heads , S, receive_head_dim]
        """
        assert x.dim() == 4, "expected input shape [B, S, send_heads, send_head_dim]"
        B, Sh, S, D = x.shape
        assert Sh == self.send_heads and D == self.send_head_dim, (
            f"input heads/dim mismatch: got ({Sh},{D}), expected "
            f"({self.send_heads},{self.send_head_dim})"
        )

        # 1) 把 heads 拼在最后一个维度： [B, S, send_heads * send_head_dim]
        x_comb = x.permute(0,2,1,3).reshape(B, S, Sh * D)

        # 2) key path
        v = self.v_embed(x_comb)            # [B, S, hidden_dim]
        v_flat = v.reshape(B * S, self.hidden_dim)
        v_hidden = self.v_mlp(v_flat)       # RegularMLP 期望 [N, hidden_dim] -> [N, hidden_dim]
        v_out_flat = self.v_out(v_hidden)   # [B*S, receive_heads * receive_head_dim]
        v_out = v_out_flat.view(B, S, self.receive_heads, self.receive_head_dim)
        v_out = v_out.permute(0,2,1,3)

        return v_out

    
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
    def __init__(self, data_files, num=None, split="train"):
        ds = load_dataset("parquet", data_files=data_files)[split]
        if num is not None:
            self.examples = list(ds)[:num]
        else:
            self.examples = list(ds)
    
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

class KVCacheDataset(Dataset):
    """Load precomputed student KV cache from disk."""
    def __init__(self, cache_dir, metadata_path=None):
        self.cache_dir = cache_dir
        self.files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
        assert len(self.files) > 0, f"No .pt files found in {cache_dir}"
        
        # Load metadata (optional)
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            print(f"Loaded metadata: {meta}")

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.cache_dir, self.files[idx])
        pkv_s = torch.load(path)  # List[(k, v)] on CPU
        return pkv_s
    
def load_target_pkv_batch(args, tokenizer, examples, device):
    """
    On-the-fly 获取 teacher 的 KV（只做一次）
    如果你也想离线 precompute teacher KV，可以扩展 precompute_kv.py
    """
    model_t = AutoModelForCausalLM.from_pretrained(
        args.model_t,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device).eval()

    all_pkv_t = []
    with torch.no_grad():
        pbar = tqdm(examples, desc="load target kv process")
        for example in pbar:
            input_ids = torch.tensor([get_input_ids_list(tokenizer, example)], device=device)
            out = model_t(input_ids=input_ids, use_cache=True)
            pkv_t = [(k.cpu(), v.cpu()) for k, v in out.past_key_values]
            all_pkv_t.append(pkv_t)
            del input_ids, out
            torch.cuda.empty_cache()
    
    return all_pkv_t

def precompute_and_save_kv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 tokenizer 和 model_s
    tokenizer = AutoTokenizer.from_pretrained(args.model_s)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(
        args.model_s,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device).eval()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "kvcache_train")
    os.makedirs(cache_dir, exist_ok=True)

    # 保存元信息
    meta = {
        "model_s": args.model_s,
        "description": "KV Cache from source model (prefill only)"
    }
    with open(os.path.join(args.output_dir, "kv_cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 加载数据集
    ds = HotPotQADataset({"train": args.train_file}, num= None, split="train")
    print(f"Loaded {len(ds)} training examples")
    
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    
    print("Starting KV Cache precomputation...")
    failed_count = 0
    success_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Prefill Progress")
        for step, batch in enumerate(pbar):
            if step < args.start_idx: continue
            if args.end_idx and step >= args.end_idx: break
            ex = batch[0]
            try:
                input_ids = torch.tensor([get_input_ids_list(tokenizer, ex)], device=device)
            
                # Prefill with student model
                out = model_s(input_ids=input_ids, use_cache=True)
                pkv = out.past_key_values  # tuple of (k, v)

                # Convert to FP16 and move to CPU
                cpu_pkv = []
                for k, v in pkv:
                    cpu_pkv.append((
                        k.cpu(),   # FP16 + CPU
                        v.cpu()
                    ))
                cpu_pkv = tuple(cpu_pkv)

                # Save as .pt file
                save_path = os.path.join(cache_dir, f"{step:06d}.pt")
                torch.save(cpu_pkv, save_path)

                success_count += 1
                if success_count % 50 == 0:
                    print(f"Saved {success_count} KV caches so far...")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at step {step}, consider reducing max_length or using smaller model.")
                    # 清理显存
                    torch.cuda.empty_cache()
                    failed_count += 1
                    continue
                else:
                    print(f"RuntimeError at step {step}: {e}")
                    failed_count += 1
            except Exception as e:
                print(f"Failed to process example {step}: {e}")
                failed_count += 1

            # Clean up
            del input_ids, out
            if step % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # Final report
    print(f"\n✅ Precomputation completed!")
    print(f"   Success: {success_count}")
    print(f"   Failed:  {failed_count}")
    print(f"   Output saved to: {cache_dir}")

    # Save stats
    stats = {
        "total_examples": len(ds),
        "success_count": success_count,
        "failed_count": failed_count,
        "output_dir": cache_dir,
        "model_s": args.model_s
    }
    with open(os.path.join(args.output_dir, "precompute_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
def train_with_cached_kv(args):
    device="cuda:7"
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------
    # 加载预计算的 student KV Cache
    # -------------------------------
    cache_dir = os.path.join(args.cache_dir, "kvcache_train")
    dataset = KVCacheDataset(cache_dir)

    grad_accum_steps = args.grad_accum_steps

    # -------------------------------
    # 加载 teacher 的 KV（一次性）
    # -------------------------------
    print("Loading target model to compute target KV...")
    train_data = HotPotQADataset({"train":args.train_file}, num=3000,split="train")
    # 只取前 len(dataset) 个样本
    examples = [train_data[i] for i in range(len(dataset))]
    target_pkvs = load_target_pkv_batch(args, tokenizer, examples, device)
    print(f"Loaded {len(target_pkvs)} target KV caches.")

    # Build MLP adapters
    N_s = 28
    N_t = 28

    head_dim_s = 128
    head_dim_t = 128
    head_num_s = 8
    head_num_t = 8
    
    k_mlps = AdapterBank(
        KAdapter(head_num_s, head_dim_s, head_num_t, head_dim_t) for _ in range(N_t)
        ).to(device)
    
    v_mlps = AdapterBank(
        KAdapter(head_num_s, head_dim_s, head_num_t, head_dim_t) for _ in range(N_t)
        ).to(device)
    
    mse_loss = nn.MSELoss()

    grad_accum_steps = args.grad_accum_steps

    
    print("Starting training k adapter...")
    optimizer = optim.Adam(k_mlps.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        k_mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}")
        for step, pkv_s in enumerate(pbar):
            
            pkv_s = [(k.to(device), v.to(device)) for k, v in pkv_s]
            pkv_t = [(k.to(device), v.to(device)) for k, v in target_pkvs[step]]

            loss = 0.0
            dist = 0.0
            cnt = 0
            for i in range(args.reuse_a_layer_start, N_t):
                s_idx = map_layer_nearest(i, N_s, N_t)
                k_s, v_s = pkv_s[s_idx]
                k_t, v_t = pkv_t[i]
                k_adapted = k_mlps(k_s, s_idx)
                loss += mse_loss(k_adapted, k_t.to(dtype=k_adapted.dtype)) 
                dist += mse_loss(k_s.to(dtype=k_adapted.dtype), k_t.to(dtype=k_adapted.dtype))
                cnt += 1

            if cnt > 0:
                loss = loss / cnt

            loss = loss / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = loss.item()
            total_loss += avg_loss * grad_accum_steps

            pbar.set_postfix({"loss": total_loss / (step+1),"dist": dist.item()/cnt})
            
            del pkv_s, pkv_t, k_adapted, loss, dist
            if step % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")

        save_path = os.path.join(args.output_dir, f"pre_mlp_k_adapters_epoch{epoch+1}.pth")
        torch.save(k_mlps.state_dict(), save_path)
        print(f"✅ Saved to {save_path}")

    
            
    print("Starting training v adapter...")
    optimizer = optim.Adam(v_mlps.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        v_mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}")

        for step, pkv_s in enumerate(pbar):
           
            pkv_s = [(k.to(device), v.to(device)) for k, v in pkv_s]
            pkv_t = [(k.to(device), v.to(device)) for k, v in target_pkvs[step]]

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
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = loss.item()
            total_loss += avg_loss * grad_accum_steps

            pbar.set_postfix({"loss": total_loss / (step + 1),"dist": dist.item()/cnt})
            
            del pkv_s, pkv_t, v_adapted, loss, dist
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")
    
        
        save_path = os.path.join(args.output_dir, f"pre_mlp_v_adapters_epoch{epoch+1}.pth")
        torch.save(v_mlps.state_dict(),save_path)
        print(f"✅ Saved to {save_path}")
        
def print_gpu_memory(device):
    allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
    reserved   = torch.cuda.memory_reserved(device)   / (1024**3)
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    used_gb = used / (1024**3)

    print(f"[{device}] GPU Memory:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Used:      {used_gb:.2f} GB")
    print(f"  Free:      {free / (1024**3):.2f} GB")
    print(f"  Max Alloc: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")

def train(args):
    # Load models and tokenizer
    device="cuda:0"
    device0="cuda:1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True,low_cpu_mem_usage=True).to(device0)
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True,low_cpu_mem_usage=True).to(device)
    model_s.eval().requires_grad_(False)
    model_t.eval().requires_grad_(False)

    # Build MLP adapters
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)

    head_dim_s = getattr(model_s.config, "head_dim", model_s.config.hidden_size // model_s.config.num_attention_heads)
    head_dim_t = getattr(model_t.config, "head_dim", model_t.config.hidden_size // model_t.config.num_attention_heads)
    head_num_s = getattr(model_s.config, "num_key_value_heads", getattr(model_s.config, "num_attention_heads", None))
    head_num_t = getattr(model_t.config, "num_key_value_heads", getattr(model_t.config, "num_attention_heads", None))
    
    k_mlps = AdapterBank(
        KAdapter(head_num_s, head_dim_s, head_num_t, head_dim_t) for _ in range(N_t)
        ).to(device)
    
    v_mlps = AdapterBank(
        KAdapter(head_num_s, head_dim_s, head_num_t, head_dim_t) for _ in range(N_t)
        ).to(device)
    
    mse_loss = nn.MSELoss()

    # Data loader
    dataset = HotPotQADataset({"train": args.train_file}, num= None, split="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    grad_accum_steps = args.grad_accum_steps

    
    print("Starting training k adapter...")
    optimizer = optim.Adam(k_mlps.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        k_mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            example = batch[0]
            input_ids = torch.tensor([get_input_ids_list(tokenizer, example)], device=device)
            input_ids0 = torch.tensor([get_input_ids_list(tokenizer, example)], device=device0)
            with torch.no_grad():
                s_out = model_s(input_ids=input_ids0, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                pkv_s = tuple(tuple(t.detach().to(device) for t in layer) for layer in s_out.past_key_values)
                pkv_t = tuple(tuple(t.detach() for t in layer) for layer in t_out.past_key_values)

            loss = 0.0
            dist = 0.0
            cnt = 0
            for i in range(args.reuse_a_layer_start, N_t):
                s_idx = map_layer_nearest(i, N_s, N_t)
                k_s, v_s = pkv_s[s_idx]
                k_t, v_t = pkv_t[i]
                k_adapted = k_mlps(k_s, s_idx)
                loss += mse_loss(k_adapted, k_t.to(dtype=k_adapted.dtype)) 
                dist += mse_loss(k_s.to(dtype=k_adapted.dtype), k_t.to(dtype=k_adapted.dtype))
                cnt += 1

            if cnt > 0:
                loss = loss / cnt

            loss = loss / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = loss.item()
            total_loss += avg_loss * grad_accum_steps

            pbar.set_postfix({"loss": avg_loss * grad_accum_steps,"dist": dist.item()/cnt})
            
            
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")

        save_path = os.path.join(args.output_dir, f"new_mlp_k_adapters_epoch{epoch+1}.pth")
        torch.save(k_mlps.state_dict(), save_path)
        print(f"✅ Saved to {save_path}")
    
    
            
    print("Starting training v adapter...")
    optimizer = optim.Adam(v_mlps.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        v_mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(pbar):
            example = batch[0]
            input_ids = torch.tensor([get_input_ids_list(tokenizer, example)], device=device)
            input_ids0 = torch.tensor([get_input_ids_list(tokenizer, example)], device=device0)
            with torch.no_grad():
                s_out = model_s(input_ids=input_ids0, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                pkv_s = tuple(tuple(t.detach().to(device) for t in layer) for layer in s_out.past_key_values)
                pkv_t = tuple(tuple(t.detach() for t in layer) for layer in t_out.past_key_values)

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
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = loss.item()
            total_loss += avg_loss * grad_accum_steps

            pbar.set_postfix({"loss": avg_loss * grad_accum_steps,"dist": dist.item()/cnt})
            
            del s_out, t_out, pkv_s, pkv_t, v_adapted, loss, dist
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")
    
        
        save_path = os.path.join(args.output_dir, f"new_mlp_v_adapters_epoch{epoch+1}.pth")
        torch.save(v_mlps.state_dict(),save_path)
        print(f"✅ Saved to {save_path}")
    

def train_dis(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load models (do NOT .to(device) here, let accelerator handle placement)
    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True)
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True)

    # freeze teacher/student params explicitly
    for p in model_s.parameters():
        p.requires_grad = False
    for p in model_t.parameters():
        p.requires_grad = False
    model_s.eval()
    model_t.eval()

    # Build MLP adapters (don't .to here)
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    head_dim_s = getattr(model_s.config, "head_dim", model_s.config.hidden_size // model_s.config.num_attention_heads)
    head_dim_t = getattr(model_t.config, "head_dim", model_t.config.hidden_size // model_t.config.num_attention_heads)
    head_num_s = getattr(model_s.config, "num_key_value_heads", getattr(model_s.config, "num_attention_heads", None))
    head_num_t = getattr(model_t.config, "num_key_value_heads", getattr(model_t.config, "num_attention_heads", None))

    k_mlps = AdapterBank(KAdapter(head_num_s, head_dim_s, head_num_t, head_dim_t) for _ in range(N_t))
    v_mlps = AdapterBank(VAdapter(head_num_s, head_dim_s, head_num_t, head_dim_t) for _ in range(N_t))  # FIXED to VAdapter

    mse_loss = nn.MSELoss()

    # Data loader (Accelerate will wrap sampler if needed)
    dataset = HotPotQADataset({"train": args.train_file}, num=5000, split="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    # Create optimizers BEFORE prepare so accelerator can wrap them
    optim_k = torch.optim.Adam(k_mlps.parameters(), lr=args.lr)
    optim_v = torch.optim.Adam(v_mlps.parameters(), lr=args.lr)

    # Prepare everything with accelerator (models, dataloader, adapters, optimizers)
    model_s, model_t, dataloader, k_mlps, v_mlps, optim_k, optim_v = accelerator.prepare(
        model_s, model_t, dataloader, k_mlps, v_mlps, optim_k, optim_v
    )

    grad_accum_steps = args.grad_accum_steps

    # Helper to optionally move teacher kv to CPU to save GPU memory:
    move_teacher_kv_to_cpu = getattr(args, "move_teacher_kv_to_cpu", False)

    # --- Train K adapters ---
    if accelerator.is_main_process:
        print("Starting training k adapter...")
    for epoch in range(args.epochs):
        k_mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"K Epoch {epoch+1}") if accelerator.is_main_process else dataloader
        for step, batch in enumerate(pbar):
            example = batch[0]
            input_ids = torch.tensor([get_input_ids_list(tokenizer, example)], device=accelerator.device)

            with torch.no_grad():
                s_out = model_s(input_ids=input_ids, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                # detach all past_key_values and optionally move teacher kv to cpu
                pkv_s = tuple(tuple(t.detach() for t in layer) for layer in s_out.past_key_values)
                if move_teacher_kv_to_cpu:
                    pkv_t = tuple(tuple(t.detach().cpu() for t in layer) for layer in t_out.past_key_values)
                else:
                    pkv_t = tuple(tuple(t.detach() for t in layer) for layer in t_out.past_key_values)

            # build loss safely
            total_layer_loss = None
            dist_val = 0.0
            cnt = 0
            for i in range(args.reuse_a_layer_start, N_t):
                s_idx = map_layer_nearest(i, N_s, N_t)
                k_s, v_s = pkv_s[s_idx]
                k_t, v_t = pkv_t[i]
                # bring teacher to current device/dtype if it was on cpu
                if k_t.device.type == "cpu":
                    k_t = k_t.to(accelerator.device, dtype=k_s.dtype)
                else:
                    k_t = k_t.to(dtype=k_s.dtype)

                k_adapted = k_mlps(k_s, s_idx)
                item = mse_loss(k_adapted, k_t)
                dist_val += mse_loss(k_s.to(dtype=k_adapted.dtype), k_t).item()
                cnt += 1
                total_layer_loss = item if total_layer_loss is None else total_layer_loss + item

            if cnt == 0:
                continue

            loss = total_layer_loss / cnt
            loss = loss / grad_accum_steps

            # use accelerator.backward for mixed precision and proper handling
            accelerator.backward(loss)

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(k_mlps.parameters(), max_norm=1.0)
                optim_k.step()
                optim_k.zero_grad()

            # gather metrics across processes
            avg_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
            total_loss += avg_loss * grad_accum_steps
            dist_g = accelerator.gather_for_metrics(torch.tensor(dist_val, device=accelerator.device)).mean().item()

            if accelerator.is_main_process:
                pbar.set_postfix({"loss": total_loss / (step + 1), "dist": dist_g / max(1, cnt)})

            # cleanup references
            del s_out, t_out, pkv_s, pkv_t, k_adapted, total_layer_loss, loss
            torch.cuda.empty_cache()

        # save on main only; unwrap model to get real weights
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")
            save_path = os.path.join(args.output_dir, f"new_mlp_k_adapters_epoch{epoch+1}.pth")
            torch.save(accelerator.unwrap_model(k_mlps).state_dict(), save_path)
            print(f"✅ Saved to {save_path}")
        accelerator.wait_for_everyone()

    # --- Train V adapters (same pattern) ---
    if accelerator.is_main_process:
        print("Starting training v adapter...")
    for epoch in range(args.epochs):
        v_mlps.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"V Epoch {epoch+1}") if accelerator.is_main_process else dataloader
        for step, batch in enumerate(pbar):
            example = batch[0]
            input_ids = torch.tensor([get_input_ids_list(tokenizer, example)], device=accelerator.device)

            with torch.no_grad():
                s_out = model_s(input_ids=input_ids, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                pkv_s = tuple(tuple(t.detach() for t in layer) for layer in s_out.past_key_values)
                if move_teacher_kv_to_cpu:
                    pkv_t = tuple(tuple(t.detach().cpu() for t in layer) for layer in t_out.past_key_values)
                else:
                    pkv_t = tuple(tuple(t.detach() for t in layer) for layer in t_out.past_key_values)

            total_layer_loss = None
            dist_val = 0.0
            cnt = 0
            for i in range(args.reuse_a_layer_start, N_t):
                s_idx = map_layer_nearest(i, N_s, N_t)
                k_s, v_s = pkv_s[s_idx]
                k_t, v_t = pkv_t[i]

                if v_t.device.type == "cpu":
                    v_t = v_t.to(accelerator.device, dtype=v_s.dtype)
                else:
                    v_t = v_t.to(dtype=v_s.dtype)

                v_adapted = v_mlps(v_s, s_idx)
                item = mse_loss(v_adapted, v_t)
                dist_val += mse_loss(v_s.to(dtype=v_adapted.dtype), v_t).item()
                cnt += 1
                total_layer_loss = item if total_layer_loss is None else total_layer_loss + item

            if cnt == 0:
                continue

            loss = total_layer_loss / cnt
            loss = loss / grad_accum_steps
            accelerator.backward(loss)

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(v_mlps.parameters(), max_norm=1.0)
                optim_v.step()
                optim_v.zero_grad()

            avg_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
            total_loss += avg_loss * grad_accum_steps
            dist_g = accelerator.gather_for_metrics(torch.tensor(dist_val, device=accelerator.device)).mean().item()

            if accelerator.is_main_process:
                pbar.set_postfix({"loss": total_loss / (step + 1), "dist": dist_g / max(1, cnt)})

            del s_out, t_out, pkv_s, pkv_t, v_adapted, total_layer_loss, loss
            torch.cuda.empty_cache()

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(pbar):.4f}")
            save_path = os.path.join(args.output_dir, f"new_mlp_v_adapters_epoch{epoch+1}.pth")
            torch.save(accelerator.unwrap_model(v_mlps).state_dict(), save_path)
            print(f"✅ Saved to {save_path}")
        accelerator.wait_for_everyone()

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
    head_num_s = 8
    head_num_t = 8

    if adpater:
        print("Using provided adapter for evaluation.")
        k_mlps = AdapterBank(
        KAdapter(head_num_s, head_dim_s, head_num_t, head_dim_t) for _ in range(N_t)
        ).to(device)
    
        v_mlps = AdapterBank(
        KAdapter(head_num_s, head_dim_s, head_num_t, head_dim_t) for _ in range(N_t)
        ).to(device)
        
        ckpt = os.path.join(args.output_dir, f"new_mlp_k_adapters_epoch{10}.pth")
        state_dict = torch.load(ckpt, map_location=device)
        k_mlps.load_state_dict(state_dict)
        k_mlps.eval()
        
        ckpt = os.path.join(args.output_dir, f"new_mlp_v_adapters_epoch{5}.pth")
        state_dict = torch.load(ckpt, map_location=device)
        v_mlps.load_state_dict(state_dict)
        v_mlps.eval()
        
    else:
        print("No adapter provided, skipping evaluation.")
        k_mlps, v_mlps = None, None

    # Load eval data
    ds = HotPotQADataset({"valid": args.valid_file}, num=100, split="valid")
    print(f"Evaluating on {args.valid_file} with {len(ds)} examples.")
    
    total_em = total_f1 = 0.0
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            pred, em, f1 = eval_one(ex, tokenizer, model_t, model_s, k_mlps ,v_mlps,args)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": em, "F1": f1})
    with open(f"wikiqa_result.log", "a") as f:
        f.write(f"Evaluation {args.reuse_a_layer_start}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")

def eval_wo_reuse(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device).eval()

    # Rebuild MLP and load weights
    N_t = len(model_t.model.layers)
    head_dim_t = getattr(model_t.config, "head_dim", ...)

    # Load eval data
    ds = HotPotQADataset({"valid": args.valid_file}, num=300, split="valid")
    print(f"Evaluating on {args.valid_file} with {len(ds)} examples.")
    
    total_em = total_f1 = 0.0
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            pred, em, f1 = evalone_wo_reuse(ex,tokenizer,model_t,args)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
    with open(f"hotpotqa_result.log", "a") as f:
        f.write(f"Evaluation {args.reuse_a_layer_start}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")
    
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
        KAdapter(8, head_dim_s, 8, head_dim_t) for _ in range(N_t)
        ).to(device)
    ckpt = os.path.join(args.output_dir, f"new_mlp_k_adapters_epoch{10}.pth")
    state_dict = torch.load(ckpt, map_location=device)
    k_mlps.load_state_dict(state_dict)
    k_mlps.eval()
        
    v_mlps = AdapterBank(
            [KAdapter(8, head_dim_s, 8, head_dim_t) for _ in range(N_t)],
            ).to(device)
    ckpt = os.path.join(args.output_dir, f"new_mlp_v_adapters_epoch{4}.pth")
    state_dict = torch.load(ckpt, map_location=device)
    v_mlps.load_state_dict(state_dict)
    v_mlps.eval()
    
    
    ds = HotPotQADataset({"train": args.valid_file}, num=None, split="train")
    dataLoader=DataLoader(ds,batch_size=1,shuffle=True,collate_fn=lambda x:x)
    with torch.no_grad():
        pbar = tqdm(dataLoader, desc="Evaluating")
        for step, batch in enumerate(pbar):
            ex=batch[0]
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
           
            print(f"k_mse:{((kloss/cnt)).item():.4f},v_mse:{((vloss/cnt).item()):.4f}")
            print(f"k_mse_adapter:{((kloss_a/cnt).item()):.4f},v_mse_adapter:{((vloss_a/cnt).item()):.4f}")
            '''
            print(f"vloss_:{(vloss_/cnt).item()}")
            print(f"tk_norm:{(tk_norm/cnt):.4f},tv_norm:{(tv_norm/cnt):.4f}")
            print(f"sk_norm:{(sk_norm/cnt):.4f},sv_norm:{(sv_norm/cnt):.4f}")
            print(f"ak_norm:{(ak_norm/cnt):.4f},av_norm:{(av_norm/cnt):.4f}")
            '''
            
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
    parser.add_argument("--pre", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="./precompute")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test:
        test(args)
    elif args.test1:
        test1(args)
    elif args.pre:
        precompute_and_save_kv(args)
    else:
        if args.train:
            train(args)
        if args.eval:
            #eval_wo_reuse(args)
            evaluate(args, adpater=args.adapter)