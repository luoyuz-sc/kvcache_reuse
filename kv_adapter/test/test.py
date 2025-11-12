from transformers import AutoModelForCausalLM, AutoTokenizer
from models import *
from torch.utils.data import DataLoader, Dataset
from data import *
from tqdm import tqdm
from utils import *
import torch.optim as optim
import torch.nn as nn
from accelerate import Accelerator

def l2norm(tensor):
    return torch.sqrt(torch.sum(tensor ** 2)) 

def test(args):
    device = "cuda:3"
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
    ckpt = os.path.join(args.output_dir, f"new_mlp_v_adapters_epoch{5}.pth")
    state_dict = torch.load(ckpt, map_location=device)
    v_mlps.load_state_dict(state_dict)
    v_mlps.eval()
    
    adapters_k = nn.ModuleList([
            ConditionalResidualAdapterPerHeadCNN(
                8, head_dim_t,
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
                8, head_dim_t,
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
    
    ds = HotPotQADataset({"valid": args.valid_file}, num=None, split="valid")
    dataLoader=DataLoader(ds,batch_size=1,shuffle=True,collate_fn=lambda x:x)
    with torch.no_grad():
        pbar = tqdm(dataLoader, desc="Evaluating")
        for step, batch in enumerate(pbar):
            if step>1000:
                break
            ex=batch[0]
            input_ids_list = ds.get_input_ids_list(tokenizer, ex)
            #input_ids_list = tokenizer("What football club plays in the area between the old tool gates: Brook Bar and Trafford bar?").input_ids
            text = tokenizer.decode(input_ids_list, skip_special_tokens=True)
            model_s_out = model_s(input_ids=torch.tensor([input_ids_list], device=device), use_cache=True)
            model_t_out = model_t(input_ids=torch.tensor([input_ids_list], device=device), use_cache=True)
            pkv_s = model_s_out.past_key_values
            pkv_t = model_t_out.past_key_values
            pkv_s = tuple(tuple(t for t in layer) for layer in pkv_s)
            pkv_t = tuple(tuple(t for t in layer) for layer in pkv_t)
            new_cache,_ = reuse_layer_with_mlp(pkv_t,pkv_s,k_mlps,v_mlps, args)
            kloss = torch.tensor(0.0, device=device)
            vloss = torch.tensor(0.0, device=device)
            kloss_a = torch.tensor(0.0, device=device)
            vloss_a = torch.tensor(0.0, device=device)
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
                cnt += 1
           
            print(f"k_mse:{((kloss/cnt)).item():.4f},v_mse:{((vloss/cnt).item()):.4f}")
            print(f"k_mse_adapter:{((kloss_a/cnt).item()):.4f},v_mse_adapter:{((vloss_a/cnt).item()):.4f}")
            pbar.set_postfix({"k_mse_adapter":((kloss_a/cnt).item()),"v_mse_adapter":(vloss_a/cnt).item()})
            

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
            input_ids_list = ds.get_input_ids_list(tokenizer, ex)
            text = tokenizer.decode(input_ids_list, skip_special_tokens=True)
            print("text:",text)
            pred = kv_bridged_generate(model_t,model_s,tokenizer,None,None,input_ids_list,args) 
            print("pred:",pred)
            
def test_attn(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).to(device0).eval()
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device).eval()
    
    model_s.config._attn_implementation="eager"
    model_t.config._attn_implementation="eager"
    
    ds = HotPotQADataset({"valid": args.valid_file}, num=100, split="valid")
    print(f"Testing on {args.valid_file} with {len(ds)} examples.")
    
    with torch.inference_mode():
        pbar = tqdm(ds, desc="Evaluating")
        for step, ex in enumerate(pbar):
            input_ids_list = ds.get_input_ids_list(tokenizer,ex)
    
            input_ids = torch.tensor([input_ids_list], device=device)
            input_ids0 = torch.tensor([input_ids_list], device=device0)

            
            tout = model_t(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            sout = model_s(
                input_ids=input_ids0,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            attn_s = sout.attentions  # tuple of length N_t; each [B, H_t, S, S]
            attn_t = tout.attentions  # tuple of length N_t; each [B, H_t, S, S]
            
            B, H, S, _ = attn_s[0].shape
            
            for i in range(28):
                attn0 = attn_s[i].reshape(B*H*S, S).to(device)
                attn1 = attn_t[i].reshape(B*H*S, S).to(device)
                diff = torch.abs(attn0 - attn1)/2
                diff = diff.sum(dim=1).mean().item()
                print(f"attn_diff in layer {i} {diff:.4f}")
                    