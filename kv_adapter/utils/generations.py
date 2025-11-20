import torch
from transformers import DynamicCache
from typing import List
from models import *
from utils.generations import *
from utils.postprocess import *
from utils.distributed import *
from config import device,device0
from torch import nn
from typing import Callable, Optional, Union
import gc
import os, time

NAN_MONITOR_ENV = os.environ.get("NAN_MONITOR", "0") == "1"

def _tensor_stats(t: torch.Tensor):
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "min": float(torch.nan_to_num(t).min().item()) if t.numel() else None,
        "max": float(torch.nan_to_num(t).max().item()) if t.numel() else None,
        "nan": bool(torch.isnan(t).any().item()),
        "inf": bool(torch.isinf(t).any().item()),
    }

def _log_nan(tag: str, tensor: torch.Tensor):
    if not (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
        return False
    stats = _tensor_stats(tensor)
    line = f"[NAN][{time.time():.0f}] {tag} stats={stats}\n"
    with open("nan_debug.log", "a") as f:
        f.write(line)
    print(line, end="")
    return True

def _maybe_check(tag: str, tensor: torch.Tensor):
    if NAN_MONITOR_ENV:
        _log_nan(tag, tensor)
 

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
    
def reuse_layer_with_mlp_recomp(a_cache, b_cache, kmlps: AdapterBank, vmlps: AdapterBank, args):
    """
    a_cache: [N_a, 2, (B, H, S, D)]  ← target KV cache
    b_cache: [N_b, 2, (B, H, S, D)]  ← source KV cache
    mlps: ModuleList of K MLPs, one for each reused layer
    Transforms selected b_cache KV states using MLP and injects into a_cache.
    """

    reuse_a_layer_start = args.reuse_a_layer_start
    ratio = args.ratio

    reuse_b_layer_list = [
        map_layer_nearest(layer_idx, len(a_cache), len(b_cache))
        for layer_idx in range(reuse_a_layer_start, len(a_cache))
    ]

    adapted_kv_list = []
    for i, b_idx in enumerate(reuse_b_layer_list):
        k,v = b_cache[b_idx]  
        ak,av = a_cache[b_idx]
        nk,nv = k,v
        B,H,S,D=k.shape
        recomp_num = int(S * ratio)
        # Apply MLP adapter
        if kmlps is not None:
            k = kmlps(k, b_idx)
            nk = []
            nk.append(ak[:,:,:recomp_num,:])
            nk.append(k[:,:,recomp_num:,:])
            nk = torch.concat(nk,dim=2)
        if vmlps is not None:
            v = vmlps(v, b_idx)
            nv = []
            nv.append(av[:,:,:recomp_num,:])
            nv.append(v[:,:,recomp_num:,:])
            nv = torch.concat(nv,dim=2)
        
        adapted_kv_list.append((nk,nv))

    # Replace from `reuse_a_layer_start` onward
    new_a_cache = list(a_cache[:reuse_a_layer_start]) + adapted_kv_list
    return tuple(new_a_cache), reuse_b_layer_list
    #return tuple(a_cache), reuse_b_layer_list
    
def reuse_layer_with_mlp_ridge(a_cache, b_cache, kmlps: AdapterBank, vmlps: AdapterBank, alpha, args):
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

    ak_list,av_list=[],[]
    bk_list,bv_list=[],[]
    for layer in range(reuse_a_layer_start):
        ak,av=a_cache[layer]
        bk,bv=b_cache[layer]
        B,H,S,D=ak.shape
        ak,av=ak.permute(0,2,1,3).reshape(B*S,H*D),av.permute(0,2,1,3).reshape(B*S,H*D)
        bk,bv=bk.permute(0,2,1,3).reshape(B*S,H*D),bv.permute(0,2,1,3).reshape(B*S,H*D)
        ak_list.append(ak)
        av_list.append(av)
        bk_list.append(bk)
        bv_list.append(bv)
        
    if reuse_a_layer_start >0:
        ak=torch.concat(ak_list,dim=0)
        av=torch.concat(av_list,dim=0)
        bk=torch.concat(bk_list,dim=0)
        bv=torch.concat(bv_list,dim=0)
        
        k_ridge=RidgeAdapter(device=device)
        k_ridge=k_ridge.fit(bk,ak,device_for_storage=device)
        v_ridge=RidgeAdapter(device=device)
        v_ridge=v_ridge.fit(bv,av,device_for_storage=device)
        
    adapted_kv_list = []
    alpha = torch.tensor(alpha,device=device)
    for i, b_idx in enumerate(reuse_b_layer_list):
        k,v = b_cache[b_idx]  
        # Apply MLP adapter
        if kmlps is not None:
            new_k = kmlps(k, b_idx)
        if vmlps is not None:
            new_v = vmlps(v, b_idx)
        if reuse_a_layer_start>0:
            new_k = (1-alpha)*new_k + alpha*(k_ridge(k))
            new_v = (1-alpha)*new_v + alpha*(v_ridge(v))
        adapted_kv_list.append((new_k,new_v))

    # Replace from `reuse_a_layer_start` onward
    new_a_cache = list(a_cache[:reuse_a_layer_start]) + adapted_kv_list
    return tuple(new_a_cache), reuse_b_layer_list
    #return tuple(a_cache), reuse_b_layer_list
    
def reuse_layer_with_mlp_res(a_cache, b_cache, kmlps: AdapterBank, vmlps: AdapterBank, kadapter,vadapter, args):
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

    adapted_kv_list=[]
    for i, b_idx in enumerate(reuse_b_layer_list):
        if i==0:
            psk,psv = b_cache[b_idx-1]
        else:
            psk,psv = adapted_kv_list[-1]
        ptk,ptv = a_cache[b_idx-1]
        k,v = b_cache[b_idx]  
        # Apply MLP adapter
        if kmlps is not None:
            new_k = kmlps(k, b_idx)
        if vmlps is not None:
            new_v = vmlps(v, b_idx)
        new_k = kadapter[b_idx](ptk,psk,k,new_k)
        new_v = vadapter[b_idx](ptv,psv,v,new_v)
        adapted_kv_list.append((new_k,new_v))

    # Replace from `reuse_a_layer_start` onward
    new_a_cache = list(a_cache[:reuse_a_layer_start]) + adapted_kv_list
    return tuple(new_a_cache), reuse_b_layer_list
    #return tuple(a_cache), reuse_b_layer_list

def map_layer_nearest(idx_t, n_layers_s, n_layers_t):
    idx_t = n_layers_s - n_layers_t + idx_t
    return idx_t

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
    #new_a_cache, reuse_b_layer_list = reuse_layer_with_mlp_res(pkv_t, pkv_s, k_mlps,v_mlps, kadapter,vadapter, args)
    new_a_cache, reuse_b_layer_list = reuse_layer_with_mlp_recomp(pkv_t, pkv_s, k_mlps,v_mlps , args)
    
    first_token = sample_next_token(t_out.logits[:,-1,:].squeeze(0), temperature, top_p)
    
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

    return postprocess(text)

@torch.no_grad()
def kv_bridged_generate_res(model_t,model_s, tok, k_mlps, v_mlps, kadapter,vadapter, input_ids_list: list[int], args):
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
    new_a_cache, reuse_b_layer_list = reuse_layer_with_mlp_res(pkv_t, pkv_s, k_mlps,v_mlps, kadapter,vadapter, args)
    
    first_token = sample_next_token(t_out.logits[:,-1,:].squeeze(0), temperature, top_p)
    
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

@torch.no_grad()
def kv_bridged_generate_ridge(model_t,model_s, tok, k_mlps, v_mlps, input_ids_list: list[int], args):
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
    
    first_token = sample_next_token(t_out.logits[:,-1,:].squeeze(0), temperature, top_p)
    
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
    return postprocess(text)

def attnr_generate(model_t,model_s, tok, input_ids_list, args):
    eos_id = tok.eos_token_id
    max_new = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    start_layer = args.reuse_a_layer_start
    

    input_ids = torch.tensor([input_ids_list], device=device)

    

    # prefill with s
    with torch.inference_mode():
        t_out = model_t(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )

        first_token = sample_next_token(t_out.logits[:,-1,:].squeeze(0), temperature, top_p)
        generated = [first_token]
        last_token = torch.tensor([[first_token]], device=device)
        
        del t_out
        
        new_a_cache = reuse_attn(model_t, model_s, input_ids_list, device, args)
        
        new_a_cache = tuple( (k.to(device),v.to(device)) for (k,v) in new_a_cache)
        '''
        pkv_t = [(k.detach(), v.detach()) for (k,v) in t_out.past_key_values]
        print("loss: ",sum([nn.MSELoss()(pkv_t[i][0],new_a_cache[i][0]) 
                for i in range(start_layer,len(pkv_t))])/(len(pkv_t)-start_layer))
        '''
        
        past = DynamicCache.from_legacy_cache(past_key_values=new_a_cache)
       
        
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
                del out, logits
                break
            # Prepare inputs for next step
            last_token = torch.tensor([[next_id]], device=device)
        
        del past, new_a_cache
        torch.cuda.empty_cache()
        gc.collect()
    
    text = tok.decode([t for t in generated if t != eos_id], skip_special_tokens=True)
    return postprocess(text)

def attnr_generate_distr(model_t,model_s, tok, input_ids_list, layer_fuser, head_fuser, args):
    eos_id = tok.eos_token_id
    max_new = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    start_layer = args.reuse_a_layer_start
    
    device_t = module_device(model_t.get_input_embeddings() or model_t.embed_tokens)
    device_s = module_device(model_s.get_input_embeddings() or model_s.embed_tokens)

    input_ids_t = torch.tensor([input_ids_list], device=device_t)
    input_ids_s = torch.tensor([input_ids_list], device=device_s)

    # prefill with s
    with torch.inference_mode():
        t_out = model_t(
            input_ids=input_ids_t,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )

        first_token = sample_next_token(t_out.logits[:,-1,:].squeeze(0), temperature, top_p)
        generated = [first_token]
        last_token = torch.tensor([[first_token]], device=device_t)
        
        del t_out
        
        new_a_cache = reuse_attn_distr_with_fuser(model_t, model_s, input_ids_list, layer_fuser, head_fuser, args)
        
        '''
        pkv_t = [(k.detach(), v.detach()) for (k,v) in t_out.past_key_values]
        print("loss: ",sum([nn.MSELoss()(pkv_t[i][0],new_a_cache[i][0]) 
                for i in range(start_layer,len(pkv_t))])/(len(pkv_t)-start_layer))
        '''
        
        past = DynamicCache.from_legacy_cache(past_key_values=new_a_cache)
       
        
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
                del out, logits
                break
            # Prepare inputs for next step
            last_token = torch.tensor([[next_id]], device=device_t)
        
        del past, new_a_cache
        torch.cuda.empty_cache()
        gc.collect()
    
    text = tok.decode([t for t in generated if t != eos_id], skip_special_tokens=True)
    return postprocess(text), text

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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
):
    _, H, _, _ = query.shape
    key_states = repeat_kv(key, H//key.shape[1])
    value_states = repeat_kv(value, H//value.shape[1])

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    _maybe_check("attn/weights_raw", attn_weights)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask # [B, H, S, S]
        _maybe_check("attn/weights_masked", attn_weights)

    # numerically stable softmax + post normalization
    attn_fp32 = attn_weights.float()
    attn_fp32 = attn_fp32 - attn_fp32.max(dim=-1, keepdim=True).values
    probs = nn.functional.softmax(attn_fp32, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = probs.sum(dim=-1, keepdim=True)
    probs = probs / (row_sum + 1e-12)
    attn_weights = probs.to(query.dtype)
    _maybe_check("attn/weights_final", attn_weights)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    _maybe_check("attn/output", attn_output)

    return attn_output, attn_weights

def find_qkv_o_projs_qwen(layer_module: nn.Module):
    """
    For Qwen-like layers (Qwen3DecoderLayer), expect:
      layer.self_attn.{q_proj,k_proj,v_proj,o_proj}
    Return (o_proj, k_proj, v_proj)
    """
    o_proj = None
    k_proj = None
    v_proj = None
    k_norm = None
    v_norm = None
    try:
        att = layer_module.self_attn
        o_proj = getattr(att, "o_proj", None) or getattr(att, "out_proj", None)
        k_proj = getattr(att, "k_proj", None)
        v_proj = getattr(att, "v_proj", None)
        k_norm = getattr(att, "k_norm", None)
        v_norm = getattr(att, "v_norm", None)
    except Exception:
        pass
    return o_proj, k_proj, v_proj, k_norm, v_norm

def apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def reuse_attn(model_t, model_s, input_ids_list, device, args):
    reuse_a_layer_start = args.reuse_a_layer_start
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    
    input_ids = torch.tensor([input_ids_list], device=device)
    input_ids0 = torch.tensor([input_ids_list], device=device0)
    
    with torch.inference_mode():
        # run both but immediately move large outputs to CPU
        tout = model_t(input_ids=input_ids, use_cache=True, output_hidden_states=True, output_attentions=False)
        sout = model_s(input_ids=input_ids0, use_cache=True, output_hidden_states=True, output_attentions=False)

        # move teacher past/hidden to CPU and detach
        pkv_t_cpu = [(k.detach().cpu(), v.detach().cpu()) for (k, v) in tout.past_key_values]
        pkv_s_cpu = [(k.detach().cpu(), v.detach().cpu()) for (k, v) in sout.past_key_values]
        hidden_states_t_cpu = tuple(h.detach().cpu() for h in tout.hidden_states)
        hidden_states_s_cpu = tuple(h.detach().cpu() for h in sout.hidden_states)


        # free GPU-held outputs quickly
        del tout, sout
        torch.cuda.empty_cache()

        # prepare CPU return buffers
        adapted_kv = [None] * N_t
        adapted_hidden = [None] * (N_t + 1)

        # prefix: for layers before reuse_start, store teacher pkv (keep on CPU)
        for li in range(min(reuse_a_layer_start + 1, N_t)):
            adapted_kv[li] = (pkv_t_cpu[li][0], pkv_t_cpu[li][1])   # CPU
            adapted_hidden[li] = hidden_states_t_cpu[li]              # CPU

        # delete pkv_t_cpu/hidden_states_cpu references if not needed
        # keep pkv_t_cpu accessible for layers we haven't computed yet (we used pkv_t_cpu[i] below)
        # but we will rely on pkv_t_cpu for those i where adapted_kv not yet computed
        
        embed_tokens_s = model_s.model.embed_tokens
        inputs_embeds_s = embed_tokens_s(input_ids0)
        position_ids_s = torch.arange(
                0, inputs_embeds_s.shape[1], device=inputs_embeds_s.device
        ).unsqueeze(0)
        
        embed_tokens_t = model_t.model.embed_tokens
        inputs_embeds_t = embed_tokens_t(input_ids)
        position_ids_t = torch.arange(
                0, inputs_embeds_t.shape[1], device=inputs_embeds_t.device
        ).unsqueeze(0)
        
        rotary_emb_s = model_s.model.rotary_emb
        coss, sins = rotary_emb_s(hidden_states_s_cpu[0].to(device0), position_ids_s)
        
        rotary_emb_t = model_t.model.rotary_emb
        cost, sint = rotary_emb_t(hidden_states_t_cpu[0].to(device), position_ids_t)
        

        # process layers sequentially, moving only required tensors to GPU
        for i in range(reuse_a_layer_start, N_t):
            s_idx = map_layer_nearest(i, N_s, N_t)
            
            s_layer = model_s.model.layers[s_idx]
            q_proj_s = s_layer.self_attn.q_proj
            q_norm_s = s_layer.self_attn.q_norm
            _, kp, vp, kn, vn = find_qkv_o_projs_qwen(s_layer)
            input_layernorm_s = s_layer.input_layernorm
            hid_s = hidden_states_s_cpu[i].to(device0)
            hid_s = input_layernorm_s(hid_s)
            
            
            input_shape = hid_s.shape[:-1]
            hidden_shape = (*input_shape, -1, 128)
            q_s = q_norm_s(q_proj_s(hid_s).view(hidden_shape)).transpose(1, 2)  # [B, H_s, S, D]
            q_s = apply_rotary_pos_emb(q_s,coss,sins)
            k_s, v_s = pkv_s_cpu[s_idx]
            k_s = k_s.to(device0)
            v_s = v_s.to(device0)
            
            
            B, _, S, D = k_s.shape
            causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=device0), diagonal=1)
            add_mask = torch.where(causal_mask, torch.tensor(-1e9, device=device0, dtype=k_s.dtype),
                    torch.tensor(0.0, device=device0, dtype=k_s.dtype))
            add_mask = add_mask.unsqueeze(0).unsqueeze(0)
            add_mask = add_mask.expand(B, 16, S, S)
            _, attn = eager_attention_forward(q_s, k_s, v_s, add_mask, 128 ** -0.5)
            attn = attn.to(device)
              
            k_cpu, v_cpu = adapted_kv[i]       # currently CPU or None

            # Move v to device for attention matmul; avoid moving k if not needed
            v = v_cpu.to(device)                   # [B, H_t, S, D] on device

            # Expand KV heads if your model requires (repeat_kv expects device tensor)
            v = repeat_kv(v, 2)                    # e.g. expand from 8->16 heads if needed

            # find layer modules on model_t (assume model_t parameters are on same device)
            t_layer = model_t.model.layers[i]
            o_proj, _, _, _, _ = find_qkv_o_projs_qwen(t_layer)

            # attn_output: [B, H_q, S, D] @> concat -> [B,S,H_q*D]
            attn_output = torch.matmul(attn, v)    # on device
            B, Hq, S, D = attn_output.shape
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, -1)
            attn_output = o_proj(attn_output)      # project to hidden (on device)

            # get the hidden input (we stored adapted_hidden[i] on CPU earlier)
            hid_cpu = adapted_hidden[i]
            hid = hid_cpu.to(device)

            # residual, layernorm, mlp
            hid = hid + attn_output
            residual = hid
            hid_ln = t_layer.post_attention_layernorm(hid)
            hid_mlp = t_layer.mlp(hid_ln)
            next_hidden = residual + hid_mlp    # on device

            # store next_hidden as CPU (detach) to avoid holding on GPU
            adapted_hidden[i + 1] = next_hidden.detach().cpu()

            # compute next layer k/v if needed
            if i + 1 < N_t:
                next_layer = model_t.model.layers[i + 1]
                _, k_proj, v_proj, k_norm, _ = find_qkv_o_projs_qwen(next_layer)
                
                input_layernorm_n = next_layer.input_layernorm
                next_hidden_n = input_layernorm_n(next_hidden)

                # compute on device then move to CPU
                next_k_flat = k_proj(next_hidden_n)   # [B,S,H_kv*D] on device
                next_v_flat = v_proj(next_hidden_n)
                # reshape as you did
                input_shape = next_k_flat.shape[:-1]
                hidden_shape = (*input_shape, -1, 128)
                next_k = k_norm(next_k_flat.view(hidden_shape)).transpose(1, 2)
                next_k = apply_rotary_pos_emb(next_k,cost,sint)
                next_v = next_v_flat.view(hidden_shape).transpose(1, 2)

                # store CPU detached versions
                adapted_kv[i + 1] = (next_k.detach().cpu(), next_v.detach().cpu())

            # clear GPU temps immediately
            del attn, v, attn_output, hid, residual, hid_ln, hid_mlp, next_hidden
            torch.cuda.empty_cache()

        # done, free CPU temporaries if any (we keep adapted_kv/adapted_hidden on CPU)
        del pkv_t_cpu
        gc.collect()

    return adapted_kv

def print_device(m: nn.Module):
    for p in m.parameters(recurse=False):
        print(p.device)

def reuse_attn_distr(model_t, model_s, input_ids_list, args):
    reuse_a_layer_start = args.reuse_a_layer_start
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    H_s = model_s.config.num_attention_heads
    H_t = model_t.config.num_attention_heads
    Hkv_s = model_s.config.num_key_value_heads
    Hkv_t = model_t.config.num_key_value_heads
    
    
    device_t_emb = module_device(model_t.get_input_embeddings() or model_t.embed_tokens)
    device_s_emb = module_device(model_s.get_input_embeddings() or model_s.embed_tokens)

    input_ids_t = torch.tensor([input_ids_list], device=device_t_emb)
    input_ids_s = torch.tensor([input_ids_list], device=device_s_emb)
    
    device_layer_t = [module_device(model_t.model.layers[i]) for i in range(N_t)]
    device_layer_s = [module_device(model_s.model.layers[i]) for i in range(N_s)]
    
    device_rotary_t = module_device(model_t.model.rotary_emb)
    device_rotary_s = module_device(model_s.model.rotary_emb)
    
    with torch.inference_mode():
        # run both but immediately move large outputs to CPU
        tout = model_t(input_ids=input_ids_t, use_cache=True, output_hidden_states=True, output_attentions=False)
        sout = model_s(input_ids=input_ids_s, use_cache=True, output_hidden_states=True, output_attentions=False)

        # move teacher past/hidden to CPU and detach
        pkv_t = [(k.detach(), v.detach()) for (k, v) in tout.past_key_values]
        pkv_s = [(k.detach(), v.detach()) for (k, v) in sout.past_key_values]
        hidden_states_t = tuple(h.detach() for h in tout.hidden_states)
        hidden_states_s = tuple(h.detach() for h in sout.hidden_states)


        # free GPU-held outputs quickly
        del tout, sout
        torch.cuda.empty_cache()

        # prepare CPU return buffers
        adapted_kv = [None] * N_t
        adapted_hidden = [None] * (N_t + 1)

        # prefix: for layers before reuse_start, store teacher pkv 
        for li in range(min(reuse_a_layer_start + 1, N_t)):
            adapted_kv[li] = (pkv_t[li][0], pkv_t[li][1])   
            adapted_hidden[li] = hidden_states_t[li]            

        # delete pkv_t_cpu/hidden_states_cpu references if not needed
        # keep pkv_t_cpu accessible for layers we haven't computed yet (we used pkv_t_cpu[i] below)
        # but we will rely on pkv_t_cpu for those i where adapted_kv not yet computed
        
        embed_tokens_s = model_s.model.embed_tokens
        inputs_embeds_s = embed_tokens_s(input_ids_s)
        position_ids_s = torch.arange(
                0, inputs_embeds_s.shape[1], device=device_rotary_s
        ).unsqueeze(0)
        
        embed_tokens_t = model_t.model.embed_tokens
        inputs_embeds_t = embed_tokens_t(input_ids_t)
        position_ids_t = torch.arange(
                0, inputs_embeds_t.shape[1], device=device_rotary_t
        ).unsqueeze(0)
        
        rotary_emb_s = model_s.model.rotary_emb
        coss, sins = rotary_emb_s(hidden_states_s[0], position_ids_s)
        
        rotary_emb_t = model_t.model.rotary_emb
        cost, sint = rotary_emb_t(hidden_states_t[0], position_ids_t)
        

        # process layers sequentially, moving only required tensors to GPU
        for i in range(reuse_a_layer_start, N_t):
            s_idx = map_layer_nearest(i, N_s, N_t)
            
            s_layer = model_s.model.layers[s_idx]
            q_proj_s = s_layer.self_attn.q_proj
            q_norm_s = s_layer.self_attn.q_norm
            input_layernorm_s = s_layer.input_layernorm
            hid_s = hidden_states_s[s_idx]
            hid_s = input_layernorm_s(hid_s)
            
            
            input_shape = hid_s.shape[:-1]
            hidden_shape = (*input_shape, -1, 128)
            q_s = q_norm_s(q_proj_s(hid_s).view(hidden_shape)).transpose(1, 2)  # [B, H_s, S, D]
            coss = coss.to(q_s.device)
            sins = sins.to(q_s.device)
            q_s = apply_rotary_pos_emb(q_s,coss,sins)
            k_s, v_s = pkv_s[s_idx]
            
            B, _, S, D = k_s.shape
            causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=device_layer_s[s_idx]), diagonal=1)
            add_mask = torch.where(causal_mask, torch.tensor(-1e9, device=device_layer_s[s_idx], dtype=k_s.dtype),
                    torch.tensor(0.0, device=device_layer_s[s_idx], dtype=k_s.dtype))
            add_mask = add_mask.unsqueeze(0).unsqueeze(0)
            add_mask = add_mask.expand(B, H_s, S, S)
            
            _, attn = eager_attention_forward(q_s, k_s, v_s, add_mask, 128 ** -0.5)
            attn = attn.reshape(B, H_s // H_t, H_t, S, S )
            attn_fused = attn.mean(dim=1) 
            attn_fused = attn_fused.to(device_layer_t[i]) # [B, H_t, S, S]

              
            k, v = adapted_kv[i]       # currently CPU or None

            # Expand KV heads if your model requires (repeat_kv expects device tensor)
            v = repeat_kv(v, H_t//Hkv_t)                    # e.g. expand from 8->16 heads if needed

            # find layer modules on model_t (assume model_t parameters are on same device)
            t_layer = model_t.model.layers[i]
            o_proj, _, _, _, _ = find_qkv_o_projs_qwen(t_layer)

            # attn_output: [B, H_q, S, D] @> concat -> [B,S,H_q*D]
            attn_output = torch.matmul(attn_fused, v)    # on device
            B, Hq, S, D = attn_output.shape
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, -1)
            attn_output = o_proj(attn_output)      # project to hidden (on device)

            # get the hidden input (we stored adapted_hidden[i] on CPU earlier)
            hid = adapted_hidden[i]

            # residual, layernorm, mlp
            hid = hid + attn_output
            residual = hid
            hid_ln = t_layer.post_attention_layernorm(hid)
            hid_mlp = t_layer.mlp(hid_ln)
            next_hidden = residual + hid_mlp    # on device

            # compute next layer k/v if needed
            if i + 1 < N_t:
                adapted_hidden[i + 1] = next_hidden.detach().to(device_layer_t[i+1])
                next_layer = model_t.model.layers[i + 1]
                _, k_proj, v_proj, k_norm, _ = find_qkv_o_projs_qwen(next_layer)
                
                input_layernorm_n = next_layer.input_layernorm
                next_hidden_n = input_layernorm_n(next_hidden)

                # compute on device then move to CPU
                next_k_flat = k_proj(next_hidden_n)   # [B,S,H_kv*D] on device
                next_v_flat = v_proj(next_hidden_n)
                # reshape as you did
                input_shape = next_k_flat.shape[:-1]
                hidden_shape = (*input_shape, -1, 128)
                next_k = k_norm(next_k_flat.view(hidden_shape)).transpose(1, 2)
                cost = cost.to(next_k.device)
                sint = sint.to(next_k.device)
                next_k = apply_rotary_pos_emb(next_k,cost,sint)
                next_v = next_v_flat.view(hidden_shape).transpose(1, 2)

                # store CPU detached versions
                adapted_kv[i + 1] = (next_k.detach(), next_v.detach())

            # clear GPU temps immediately
            del attn, v, attn_output, hid, residual, hid_ln, hid_mlp, next_hidden
            torch.cuda.empty_cache()

        # done, free CPU temporaries if any (we keep adapted_kv/adapted_hidden on CPU)
        del pkv_t
        gc.collect()

    return adapted_kv

def reuse_attn_distr_with_fuser(model_t, model_s, input_ids_list, 
                                layer_fuser:LayerAttentionFuser, head_fuser:HeadAttentionFuser, args):
    reuse_a_layer_start = args.reuse_a_layer_start
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    H_s = model_s.config.num_attention_heads
    H_t = model_t.config.num_attention_heads
    Hkv_s = model_s.config.num_key_value_heads
    Hkv_t = model_t.config.num_key_value_heads
    
    
    device_t_emb = module_device(model_t.get_input_embeddings() or model_t.embed_tokens)
    device_s_emb = module_device(model_s.get_input_embeddings() or model_s.embed_tokens)

    input_ids_t = torch.tensor([input_ids_list], device=device_t_emb)
    input_ids_s = torch.tensor([input_ids_list], device=device_s_emb)
    
    device_layer_t = [module_device(model_t.model.layers[i]) for i in range(N_t)]
    device_layer_s = [module_device(model_s.model.layers[i]) for i in range(N_s)]
    
    device_rotary_t = module_device(model_t.model.rotary_emb)
    device_rotary_s = module_device(model_s.model.rotary_emb)
    
    # run both but immediately move large outputs to CPU
    tout = model_t(input_ids=input_ids_t, use_cache=True, output_hidden_states=True, output_attentions=False)
    sout = model_s(input_ids=input_ids_s, use_cache=True, output_hidden_states=True, output_attentions=False)
    

    # move teacher past/hidden to CPU and detach
    pkv_t = [(k.detach(), v.detach()) for (k, v) in tout.past_key_values]
    pkv_s = [(k.detach(), v.detach()) for (k, v) in sout.past_key_values]
    hidden_states_t = tuple(h.detach().cpu() for h in tout.hidden_states)
    hidden_states_s = tuple(h.detach().cpu() for h in sout.hidden_states)

    # free GPU-held outputs quickly
    del tout, sout
    torch.cuda.empty_cache()

    # prepare CPU return buffers
    adapted_kv = [None] * N_t
    adapted_hidden = [None] * (N_t + 1)

    # prefix: for layers before reuse_start, store teacher pkv 
    for li in range(min(reuse_a_layer_start + 1, N_t)):
        adapted_kv[li] = (pkv_t[li][0], pkv_t[li][1])   
        adapted_hidden[li] = hidden_states_t[li]            

    # delete pkv_t_cpu/hidden_states_cpu references if not needed
    # keep pkv_t_cpu accessible for layers we haven't computed yet (we used pkv_t_cpu[i] below)
    # but we will rely on pkv_t_cpu for those i where adapted_kv not yet computed
    
    with torch.no_grad():
        embed_tokens_s = model_s.model.embed_tokens
        inputs_embeds_s = embed_tokens_s(input_ids_s)
        position_ids_s = torch.arange(
                0, inputs_embeds_s.shape[1], device=device_rotary_s
        ).unsqueeze(0)
        
        embed_tokens_t = model_t.model.embed_tokens
        inputs_embeds_t = embed_tokens_t(input_ids_t)
        position_ids_t = torch.arange(
                0, inputs_embeds_t.shape[1], device=device_rotary_t
        ).unsqueeze(0)
        
        rotary_emb_s = model_s.model.rotary_emb
        coss, sins = rotary_emb_s(hidden_states_s[0].to(device_rotary_s), position_ids_s)
        
        rotary_emb_t = model_t.model.rotary_emb
        cost, sint = rotary_emb_t(hidden_states_t[0].to(device_rotary_t), position_ids_t)
        

    # process layers sequentially, moving only required tensors to GPU
    for i in range(reuse_a_layer_start, N_t):
        group = layer_fuser.group_indices[i]
        #group = [i+N_s-N_t]
        
        group_attn = []
        for s_idx in group:
            with torch.no_grad():
                s_layer = model_s.model.layers[s_idx]
                q_proj_s = s_layer.self_attn.q_proj
                q_norm_s = s_layer.self_attn.q_norm
                input_layernorm_s = s_layer.input_layernorm
                hid_s = hidden_states_s[s_idx]
                hid_s = input_layernorm_s(hid_s)
                
                
                input_shape = hid_s.shape[:-1]
                hidden_shape = (*input_shape, -1, 128)
                q_s = q_norm_s(q_proj_s(hid_s).view(hidden_shape)).transpose(1, 2)  # [B, H_s, S, D]
                coss = coss.to(q_s.device)
                sins = sins.to(q_s.device)
                q_s = apply_rotary_pos_emb(q_s,coss,sins)
                k_s, v_s = pkv_s[s_idx]
                
                B, _, S, D = k_s.shape
                causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=device_layer_s[s_idx]), diagonal=1)
                add_mask = torch.where(causal_mask, torch.tensor(-1e9, device=device_layer_s[s_idx], dtype=k_s.dtype),
                        torch.tensor(0.0, device=device_layer_s[s_idx], dtype=k_s.dtype))
                add_mask = add_mask.unsqueeze(0).unsqueeze(0)
                add_mask = add_mask.expand(B, H_s, S, S)
                
                _, attn = eager_attention_forward(q_s, k_s, v_s, add_mask, 128 ** -0.5)
                
            attn = head_fuser(attn, s_idx) # [B, H_t, S, S]
            group_attn.append(attn.to(device_layer_t[i]))
        attn_fused = torch.stack(group_attn, dim=0)
        attn_fused = layer_fuser(attn_fused, i) # [B, H_t, S, S]
        
        #attn_fused = group_attn[0]
            
        k, v = adapted_kv[i]       # currently CPU or None

        # Expand KV heads if your model requires (repeat_kv expects device tensor)
        v = repeat_kv(v, H_t//Hkv_t)                    # e.g. expand from 8->16 heads if needed

        # find layer modules on model_t (assume model_t parameters are on same device)
        t_layer = model_t.model.layers[i]
        o_proj, _, _, _, _ = find_qkv_o_projs_qwen(t_layer)

        # attn_output: [B, H_q, S, D] @> concat -> [B,S,H_q*D]
        attn_output = torch.matmul(attn_fused, v)    # on device

        B, Hq, S, D = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, -1)
        attn_output = o_proj(attn_output)      # project to hidden (on device)

        # get the hidden input 
        hid = adapted_hidden[i].to(attn_output.device)

        # residual, layernorm, mlp
        hid = hid + attn_output
        residual = hid
        hid_ln = t_layer.post_attention_layernorm(hid)
        hid_mlp = t_layer.mlp(hid_ln)
        next_hidden = residual + hid_mlp    # on device

        # compute next layer k/v if needed
        if i + 1 < N_t:
            adapted_hidden[i + 1] = next_hidden.to(device_layer_t[i+1])
            next_layer = model_t.model.layers[i + 1]
            _, k_proj, v_proj, k_norm, _ = find_qkv_o_projs_qwen(next_layer)
            
            input_layernorm_n = next_layer.input_layernorm
            next_hidden_n = input_layernorm_n(next_hidden)

            # compute on device then move to CPU
            next_k_flat = k_proj(next_hidden_n)   # [B,S,H_kv*D] on device
            next_v_flat = v_proj(next_hidden_n)
            # reshape as you did
            input_shape = next_k_flat.shape[:-1]
            hidden_shape = (*input_shape, -1, 128)
            next_k = k_norm(next_k_flat.view(hidden_shape)).transpose(1, 2)
            cost = cost.to(next_k.device)
            sint = sint.to(next_k.device)
            next_k = apply_rotary_pos_emb(next_k,cost,sint)

            next_v = next_v_flat.view(hidden_shape).transpose(1, 2)


            adapted_kv[i + 1] = (next_k, next_v)

        # clear GPU temps immediately
        del attn, v, attn_output, hid, residual, hid_ln, hid_mlp, next_hidden
        torch.cuda.empty_cache()

        # done, free CPU temporaries if any (we keep adapted_kv/adapted_hidden on CPU)
        gc.collect()

    return adapted_kv
    
def reuse_attn_distr_with_fuser_accelerated(model_t, model_s, input_ids_list, 
                                layer_fuser:LayerAttentionFuser, head_fuser:HeadAttentionFuser, args):
    reuse_a_layer_start = args.reuse_a_layer_start
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    H_s = model_s.config.num_attention_heads
    H_t = model_t.config.num_attention_heads
    Hkv_s = model_s.config.num_key_value_heads
    Hkv_t = model_t.config.num_key_value_heads
    
    
    device_t_emb = module_device(model_t.get_input_embeddings() or model_t.embed_tokens)
    device_s_emb = module_device(model_s.get_input_embeddings() or model_s.embed_tokens)

    input_ids_t = torch.tensor([input_ids_list], device=device_t_emb)
    input_ids_s = torch.tensor([input_ids_list], device=device_s_emb)
    
    device_layer_t = [module_device(model_t.model.layers[i]) for i in range(N_t)]
    device_layer_s = [module_device(model_s.model.layers[i]) for i in range(N_s)]
    
    device_rotary_t = module_device(model_t.model.rotary_emb)
    device_rotary_s = module_device(model_s.model.rotary_emb)
    
    model_s.config._attn_implementation = "eager"
    
    # run both but immediately move large outputs to CPU
    with torch.inference_mode():
        tout = model_t(input_ids=input_ids_t, use_cache=True, output_hidden_states=True, output_attentions=False)
        sout = model_s(input_ids=input_ids_s, use_cache=True, output_hidden_states=True, output_attentions=True)
    

    # move teacher past/hidden to CPU and detach
    pkv_t = [(k.detach(), v.detach()) for (k, v) in tout.past_key_values]
    hidden_states_t = tuple(h.detach().cpu() for h in tout.hidden_states)
    hidden_states_s = tuple(h.detach().cpu() for h in sout.hidden_states)
    attn_src = [attn.detach().cpu() for attn in sout.attentions]

    # free GPU-held outputs quickly
    del tout, sout

    # prepare CPU return buffers
    adapted_kv = [None] * N_t
    adapted_hidden = [None] * (N_t + 1)

    # prefix: for layers before reuse_start, store teacher pkv 
    for li in range(min(reuse_a_layer_start + 1, N_t)):
        adapted_kv[li] = (pkv_t[li][0], pkv_t[li][1])   
        adapted_hidden[li] = hidden_states_t[li]            

    # delete pkv_t_cpu/hidden_states_cpu references if not needed
    # keep pkv_t_cpu accessible for layers we haven't computed yet (we used pkv_t_cpu[i] below)
    # but we will rely on pkv_t_cpu for those i where adapted_kv not yet computed
    
    with torch.no_grad():
        embed_tokens_s = model_s.model.embed_tokens
        inputs_embeds_s = embed_tokens_s(input_ids_s)
        position_ids_s = torch.arange(
                0, inputs_embeds_s.shape[1], device=device_rotary_s
        ).unsqueeze(0)
        
        embed_tokens_t = model_t.model.embed_tokens
        inputs_embeds_t = embed_tokens_t(input_ids_t)
        position_ids_t = torch.arange(
                0, inputs_embeds_t.shape[1], device=device_rotary_t
        ).unsqueeze(0)
        
        rotary_emb_s = model_s.model.rotary_emb
        coss, sins = rotary_emb_s(hidden_states_s[0].to(device_rotary_s), position_ids_s)
        
        rotary_emb_t = model_t.model.rotary_emb
        cost, sint = rotary_emb_t(hidden_states_t[0].to(device_rotary_t), position_ids_t)
        

    device_layer_fuser = next(layer_fuser.parameters()).device
    device_head_fuser = next(head_fuser.parameters()).device
    
    # process layers sequentially, moving only required tensors to GPU
    for i in range(reuse_a_layer_start, N_t):
        group = layer_fuser.group_indices[i]
        
        group_attn = []
        for s_idx in group:
            attn = attn_src[s_idx].to(device_head_fuser)
            attn = head_fuser(attn, s_idx) # [B, H_t, S, S]
            group_attn.append(attn.to(device_layer_t[i]))
        attn_fused = torch.stack(group_attn, dim=0).to(device_layer_fuser)
        _maybe_check(f"reuse_fuser_acc/group_stack/{i}", attn_fused)
        attn_fused = layer_fuser(attn_fused, i) # [B, H_t, S, S]
        _maybe_check(f"reuse_fuser_acc/layer_fused/{i}", attn_fused)
            
        k, v = adapted_kv[i]       # currently CPU or None

        # Expand KV heads if your model requires (repeat_kv expects device tensor)
        v = repeat_kv(v, H_t//Hkv_t)                    # e.g. expand from 8->16 heads if needed

        # find layer modules on model_t (assume model_t parameters are on same device)
        t_layer = model_t.model.layers[i]
        o_proj, _, _, _, _ = find_qkv_o_projs_qwen(t_layer)

        # attn_output: [B, H_q, S, D] @> concat -> [B,S,H_q*D]
        attn_fused = attn_fused.to(v.device)
        attn_output = torch.matmul(attn_fused, v)    # on device
        _maybe_check(f"reuse_fuser_acc/attn_output/{i}", attn_output)
        B, Hq, S, D = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, -1)
        attn_output = o_proj(attn_output)      # project to hidden (on device)

        # get the hidden input 
        hid = adapted_hidden[i].to(attn_output.device)

        # residual, layernorm, mlp
        hid = hid + attn_output
        residual = hid
        hid_ln = t_layer.post_attention_layernorm(hid)
        hid_mlp = t_layer.mlp(hid_ln)
        next_hidden = residual + hid_mlp    # on device

        # compute next layer k/v if needed
        if i + 1 < N_t:
            adapted_hidden[i + 1] = next_hidden.to(device_layer_t[i+1])
            next_layer = model_t.model.layers[i + 1]
            _, k_proj, v_proj, k_norm, _ = find_qkv_o_projs_qwen(next_layer)
            
            input_layernorm_n = next_layer.input_layernorm
            next_hidden_n = input_layernorm_n(next_hidden)

            # compute on device then move to CPU
            next_k_flat = k_proj(next_hidden_n)   # [B,S,H_kv*D] on device
            next_v_flat = v_proj(next_hidden_n)
            # reshape as you did
            input_shape = next_k_flat.shape[:-1]
            hidden_shape = (*input_shape, -1, 128)
            next_k = k_norm(next_k_flat.view(hidden_shape)).transpose(1, 2)
            cost = cost.to(next_k.device)
            sint = sint.to(next_k.device)
            next_k = apply_rotary_pos_emb(next_k,cost,sint)
            _maybe_check(f"reuse_fuser_acc/next_k/{i}", next_k)
            next_v = next_v_flat.view(hidden_shape).transpose(1, 2)
            _maybe_check(f"reuse_fuser_acc/next_v/{i}", next_v)

            adapted_kv[i + 1] = (next_k, next_v)

        # clear GPU temps immediately
        del attn, v, attn_output, hid, residual, hid_ln, hid_mlp, next_hidden

        # done, free CPU temporaries if any (we keep adapted_kv/adapted_hidden on CPU)
        gc.collect()

    return adapted_kv
    
    

    