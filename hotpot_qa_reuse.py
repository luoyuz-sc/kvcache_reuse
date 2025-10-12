# hotpot_qa.py
import argparse, re, string, math, random
from typing import Dict, List, Tuple, Optional
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import json

from finetune.kv_client import RemoteKVClient

# ---------------- Metrics (Hotpot-style EM/F1) ----------------
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

# ---------------- Prompting ----------------
def format_context(example: Dict) -> str:
    """Compact, readable multi-hop context."""
    titles = example["context"]["title"]
    sents = example["context"]["sentences"]
    sections = []
    for t, ss in list(zip(titles, sents)):
        snippet = " ".join(ss)
        sections.append(f"- {t}: {snippet}")
    return "\n".join(sections)

INSTRUCT_HEADER = (
    "You are a precise question answering assistant. Use the CONTEXT to answer the QUESTION.\n"
    "Return the **shortest** possible answer (e.g., single entity or 'yes'/'no'); no explanation.\n"
)

def get_input_ids_list(tokenizer, example: Dict) -> list[int]:
    ctx = format_context(example)
    q = example["question"]
    sys = INSTRUCT_HEADER.strip()
    prompt_str = INSTRUCT_HEADER + f"\nCONTEXT:\n{ctx}\n\nQUESTION: {q}\nANSWER:"
    return tokenizer(prompt_str).input_ids

def postprocess(generated: str) -> str:
    # Take only the first non-empty line; trim common prefixes.
    for line in generated.strip().splitlines():
        ans = line.strip()
        if not ans:
            continue
        ans = re.sub(r"^(Answer|A|Assistant|Final Answer)\s*[:\-]\s*", "", ans, flags=re.I)
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


def reuse_layer(a_cache, b_cache, args):
    """
    a_cache: [32, 2, [1, seqlen, 8, 128]]
    b_cache: [80, 2, [1, seqlen, 8, 128]]

    return: new_a_cache, reuse_b_layer_list
    note that different layers in a_cache and b_cache may be on different devices
    and new_a_cache should keep the same device placement as a_cache
    """
    print(f"a_cache: {len(a_cache)} layers, b_cache: {len(b_cache)} layers")
    reuse_a_layer_start = args.reuse_a_layer_start
    a_kv_cache_list = [(a_cache[layer_idx][0].to("cuda:0"), a_cache[layer_idx][1].to("cuda:0")) for layer_idx in range(reuse_a_layer_start, len(a_cache))]
    b_kv_cache_list = [(b_cache[layer_idx][0].to("cuda:0"), b_cache[layer_idx][1].to("cuda:0")) for layer_idx in range(len(b_cache))]

    dp = [[0.0]*(len(b_cache)+1) for _ in range(len(a_kv_cache_list)+1)]
    choice = [[-1]*(len(b_cache)+1) for _ in range(len(a_kv_cache_list)+1)]
    for i in range(1, len(a_kv_cache_list)+1):
        dp[i][0] = math.inf
        for j in range(1, len(b_kv_cache_list)+1):
            a_k_cache, a_v_cache = a_kv_cache_list[i-1]
            b_k_cache, b_v_cache = b_kv_cache_list[j-1]
            abs_diff = (a_k_cache - b_k_cache).abs().sum() + (a_v_cache - b_v_cache).abs().sum()
            if dp[i-1][j-1] + abs_diff < dp[i][j-1]:
                dp[i][j] = dp[i-1][j-1] + abs_diff
                choice[i][j] = j-1
            else:
                dp[i][j] = dp[i][j-1]
                choice[i][j] = -1

    reuse_b_layer_list = []
    reused_a_cache = []
    i, j = len(a_kv_cache_list), len(b_kv_cache_list)
    while i > 0 and j > 0:
        if choice[i][j] != -1:
            reuse_b_layer_list.append(choice[i][j])
            reused_a_cache.append((
                b_kv_cache_list[choice[i][j]][0],
                b_kv_cache_list[choice[i][j]][1]
            ))
            i -= 1
            j -= 1
        else:
            j -= 1
    if i != 0:
        raise ValueError("Not all layers in A are reused, please check the code.")
    # reverse the list to make it in the original order
    reuse_b_layer_list.reverse()
    reused_a_cache.reverse()
    for i in range(reuse_a_layer_start, len(a_cache)):
        reused_a_cache[i - reuse_a_layer_start] = (
            reused_a_cache[i - reuse_a_layer_start][0].to(a_cache[i][0].device),
            reused_a_cache[i - reuse_a_layer_start][1].to(a_cache[i][1].device)
        )
    # print(f"Optimal reuse_b_layer_list: {reuse_b_layer_list} with total abs diff {dp[len(a_kv_cache_list)][len(b_kv_cache_list)]}")
    write_log(args, f"Optimal reuse_b_layer_list: {reuse_b_layer_list} with total abs diff {dp[len(a_kv_cache_list)][len(b_kv_cache_list)]}")

    # Greedy Policy
    # reuse_b_layer_list = []
    # reused_a_cache = []
    # for layer_idx in range(reuse_a_layer_start, len(a_cache)):
    #     a_k_cache = a_cache[layer_idx][0].to("cuda:0")
    #     a_v_cache = a_cache[layer_idx][1].to("cuda:0")
    #     min_abs_diff = math.inf
    #     reuse_b_layer_idx = -1
    #     for b_idx in range(0, len(b_cache)):
    #         if b_idx in reuse_b_layer_list:
    #             continue
    #         b_k_cache = b_cache[b_idx][0].to("cuda:0")
    #         b_v_cache = b_cache[b_idx][1].to("cuda:0")
    #         abs_diff = (a_k_cache - b_k_cache).abs().sum() + (a_v_cache - b_v_cache).abs().sum()
    #         if abs_diff < min_abs_diff:
    #             min_abs_diff = abs_diff
    #             reuse_b_layer_idx = b_idx
    #     print(f"Layer {layer_idx} in A reuses layer {reuse_b_layer_idx} in B with abs diff {min_abs_diff}")
    #     reuse_b_layer_list.append(reuse_b_layer_idx)
    #     reused_a_cache.append( (b_cache[reuse_b_layer_idx][0].to(a_cache[layer_idx][0].device), b_cache[reuse_b_layer_idx][1].to(a_cache[layer_idx][1].device)) )

    new_a_cache = a_cache[:reuse_a_layer_start] + tuple(reused_a_cache)
    return new_a_cache, reuse_b_layer_list


@torch.no_grad()
def kv_bridged_generate(model, tok, input_ids_list: list[int], args):
    """return: generated string"""

    eos_id = tok.eos_token_id
    max_new = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p

    device = next(model.parameters()).device
    input_ids = torch.tensor([input_ids_list], device=device)

    # prefill with A
    a_out = model(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    )
    pkv8 = tuple(tuple(t for t in layer) for layer in a_out.past_key_values)

    # remote prefill
    client = RemoteKVClient("http://localhost:8000")
    pkv70, logits70 = client.prefill(input_ids[0], return_dtype=str(model.dtype).split(".")[-1], return_logits=True)
    
    # substitue layer_a of a_cache with layer_b of b_cache
    new_a_cache, reuse_b_layer_list = reuse_layer(pkv8, pkv70, args)
    first_token = sample_next_token(logits70.squeeze(0), temperature, top_p)

    first_token = sample_next_token(a_out.logits[:, -1, :].squeeze(0), temperature, top_p)
    past = DynamicCache.from_legacy_cache(past_key_values=new_a_cache)
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
    print(f"generated: {generated}")
    print(f"text: {text}")
    return postprocess(text)

def eval_one(example: Dict, tok, model, args) -> Tuple[str, float, float, str]:
    input_ids_list = get_input_ids_list(tok, example)
    pred = kv_bridged_generate(model, tok, input_ids_list, args)
    gold = example["answer"]
    return pred, exact_match(pred, gold), f1_score(pred, gold)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_log(args, message: str):
    with open(args.log_file, "a") as f:
        f.write(message + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--chat_template_path", type=str, default=None)
    parser.add_argument("--reuse_a_layer_start", type=int, default=0, help="which layer in model A to start reusing")
    args = parser.parse_args()
    
    set_seed(0)
    
    write_log(args, "Loading dataset…")
    ds = [json.loads(line) for line in open(args.jsonl_path, "r")]

    write_log(args, f"Loading tokenizer from model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template_path is not None:
        with open(args.chat_template_path, "r") as f:
            tok.chat_template = f.read()

    write_log(args, f"Loading model (decoder): {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    n = len(ds)
    ems, f1s = [], []
    for i in range(n):
        ex = ds[i]
        pred, em, f1 = eval_one(ex, tok, model, args)
        ems.append(em); f1s.append(f1)
        
        write_log(args, f"[{i+1:4d}/{n}] EM={int(em)} F1={f1:.3f}  |  Q: {ex['question']}\n    Pred: {pred}  |  Gold: {ex['answer']}")

    def write_result():
        with open(args.result_file, "a") as f:
            f.write(f"Model A (decoder): {args.model}\n")
            # f.write(f"Model B (prefill): {args.model_b} with layer {reuse_b_layer_list} transferred\n")
            f.write(f"Samples: {n}\n")
            f.write(f"EM: {sum(ems)/n:.3f}\n")
            f.write(f"F1: {sum(f1s)/n:.3f}\n")
            f.write("\n")
    write_result()

if __name__ == "__main__":
    main()
