from transformers import AutoModelForCausalLM, AutoTokenizer
from models import *
from torch.utils.data import DataLoader, Dataset
from data import *
from tqdm import tqdm
from utils.generations import *
import torch.optim as optim
import torch.nn as nn
from accelerate import Accelerator
from config import device,device0

def evaluate(args, adapter=True):
    
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

    if adapter:
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
    ds = HotPotQADataset({"valid": args.valid_file}, num=1000, split="valid")
    print(f"Evaluating on {args.valid_file} with {len(ds)} examples.")
    
    total_em = total_f1 = 0.0
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            pred, em, f1 = ds.eval_one(ex, tokenizer, model_t, model_s, k_mlps , v_mlps,args)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
    with open(f"qa_result.log", "a") as f:
        f.write(f"Evaluation {args.reuse_a_layer_start}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")
        
def evaluate_ridge(args, adapter=True):
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

    if adapter:
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
    ds = HotPotQADataset({"valid": args.valid_file}, num=1000, split="valid")
    print(f"Evaluating on {args.valid_file} with {len(ds)} examples.")
    
    total_em = total_f1 = 0.0
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            pred, em, f1 = ds.eval_one(ex, tokenizer, model_t, model_s, k_mlps ,v_mlps,args)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
    with open(f"wikiqa_result.log", "a") as f:
        f.write(f"Evaluation {args.reuse_a_layer_start}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")
        
def evaluate_res(args, adapter=True):
    
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

    if adapter:
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
        
        adapters_k = nn.ModuleList([
            ConditionalResidualAdapterPerHeadCNN(
                head_num_t, head_dim_t,
                cnn_channels=getattr(args, "cnn_channels", 64),
                cond_dim=getattr(args, "cond_dim", 64),
                n_cnn_layers=getattr(args, "n_cnn_layers", 2),
                cnn_kernel=getattr(args, "cnn_kernel", 3),
                pred_hidden=getattr(args, "pred_hidden", 256),
                dropout=getattr(args, "dropout", 0.1),
            ) for _ in range(N_t)
        ]).to(device)
        
        adapters_v = nn.ModuleList([
            ConditionalResidualAdapterPerHeadCNN(
                head_num_t, head_dim_t,
                cnn_channels=getattr(args, "cnn_channels", 64),
                cond_dim=getattr(args, "cond_dim", 64),
                n_cnn_layers=getattr(args, "n_cnn_layers", 2),
                cnn_kernel=getattr(args, "cnn_kernel", 3),
                pred_hidden=getattr(args, "pred_hidden", 256),
                dropout=getattr(args, "dropout", 0.1),
            ) for _ in range(N_t)
        ]).to(device)
        
        ckpt = os.path.join(args.output_dir, f"perhead_cond_k_adapters_cnn_epoch3.pth")
        state_dict = torch.load(ckpt, map_location=device)
        adapters_k.load_state_dict(state_dict)
        adapters_k.eval()
        
        ckpt = os.path.join(args.output_dir, f"perhead_cond_v_adapters_cnn_epoch3.pth")
        state_dict = torch.load(ckpt, map_location=device)
        adapters_v.load_state_dict(state_dict)
        adapters_v.eval()
        
    else:
        print("No adapter provided, skipping evaluation.")
        k_mlps, v_mlps = None, None
        
        
    

    # Load eval data
    ds = HotPotQADataset({"valid": args.valid_file}, num=1000, split="valid")
    print(f"Evaluating on {args.valid_file} with {len(ds)} examples.")
    
    total_em = total_f1 = 0.0
    with torch.no_grad():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            pred, em, f1 = ds.eval_one_res(ex, tokenizer, model_t, model_s, k_mlps , v_mlps, adapters_k, adapters_v,args)
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
    with open(f"qa_result.log", "a") as f:
        f.write(f"Evaluation {args.reuse_a_layer_start}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")
        
        
        
def find_qkv_o_projs_qwen(layer_module: nn.Module):
    """
    For Qwen-like layers (Qwen3DecoderLayer), expect:
      layer.self_attn.{q_proj,k_proj,v_proj,o_proj}
    Return (o_proj, k_proj, v_proj)
    """
    o_proj = None
    k_proj = None
    v_proj = None
    try:
        att = layer_module.self_attn
        o_proj = getattr(att, "o_proj", None) or getattr(att, "out_proj", None)
        k_proj = getattr(att, "k_proj", None)
        v_proj = getattr(att, "v_proj", None)
    except Exception:
        pass
    return o_proj, k_proj, v_proj

def reduce_teacher_heads_to_student(attn_t: torch.Tensor, H_s: int) -> torch.Tensor:
    """
    Reduce teacher attention heads (B, H_t, S, S) into H_s heads by grouping/averaging.
    """
    B, H_t, S, _ = attn_t.shape
    if H_t == H_s:
        return attn_t
    groups = [[] for _ in range(H_s)]
    for i in range(H_t):
        gid = int(i * H_s / H_t)
        if gid >= H_s:
            gid = H_s - 1
        groups[gid].append(i)
    out = attn_t.new_zeros((B, H_s, S, S))
    for gid, idxs in enumerate(groups):
        # mean across the chosen teacher heads for the group
        out[:, gid, :, :] = attn_t[:, idxs, :, :].mean(dim=1)
    return out


def evaluate_attn(args, adapter=True):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).to(device0).eval()
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device).eval()


    # Load eval data
    ds = HotPotQADataset({"valid": args.valid_file}, num=1000, split="valid")
    print(f"Evaluating on {args.valid_file} with {len(ds)} examples.")
    
    total_em = total_f1 = 0.0
    with torch.inference_mode():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            input_id_list = ds.get_input_ids_list(tokenizer,ex)
            pred = attnr_generate(model_t,model_s, tokenizer,  input_id_list, args)
            #pred = pure_generate(model_t, tokenizer, input_id_list, args)
            gold = ex["answer"]
            with open(f"./log/qa_result.log", "a") as f:
                f.write(f"Q: {ex['question']}\nA: {pred}\nG: {gold}\n")
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
    
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
            
    with open(f"attnr_result.log", "a") as f:
        f.write(f"Evaluation {args.reuse_a_layer_start}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")


def evaluate_attn_distr(args, adapter=True):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(
        args.model_s, trust_remote_code=True, device_map="auto", 
        torch_dtype=torch.float16, offload_folder="./tmp/offload_s")
    model_t = AutoModelForCausalLM.from_pretrained(
        args.model_t, trust_remote_code=True, device_map="auto", 
        torch_dtype=torch.float16, offload_folder="./tmp/offload_t")

    # Load eval data
    ds = HotPotQADataset({"valid": args.valid_file}, num=1000, split="valid")
    print(f"Evaluating on {args.valid_file} with {len(ds)} examples.")
    
    total_em = total_f1 = 0.0
    with torch.inference_mode():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            input_id_list = ds.get_input_ids_list(tokenizer,ex)
            pred = attnr_generate_distr(model_t,model_s, tokenizer,  input_id_list, args)
            #pred = pure_generate(model_t, tokenizer, input_id_list, args)
            gold = ex["answer"]
            with open(f"./log/qa_result.log", "a") as f:
                f.write(f"Q: {ex['question']}\nA: {pred}\nG: {gold}\n")
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
    
            total_em += em
            total_f1 += f1
            avg_em = total_em / (step + 1)
            avg_f1 = total_f1 / (step + 1)
            pbar.set_postfix({"EM": avg_em, "F1": avg_f1})
            
    with open(f"attnr_result.log", "a") as f:
        f.write(f"Evaluation {args.reuse_a_layer_start}: EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")

