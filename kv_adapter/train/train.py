from transformers import AutoModelForCausalLM, AutoTokenizer
from models import *
from torch.utils.data import DataLoader, Dataset
from data import *
from tqdm import tqdm
from utils import *
import torch.optim as optim
import torch.nn as nn
from accelerate import Accelerator


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
    dataset = HotPotQADataset({"train": args.train_file}, num= 10000, split="train")
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
            input_ids = torch.tensor([dataset.get_input_ids_list(tokenizer, example)], device=device)
            input_ids0 = torch.tensor([dataset.get_input_ids_list(tokenizer, example)], device=device0)
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

        save_path = os.path.join(args.output_dir, f"new_new_mlp_k_adapters_epoch{epoch+1}.pth")
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
            input_ids = torch.tensor([dataset.get_input_ids_list(tokenizer, example)], device=device)
            input_ids0 = torch.tensor([dataset.get_input_ids_list(tokenizer, example)], device=device0)
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
    
        
        save_path = os.path.join(args.output_dir, f"new_new_mlp_v_adapters_epoch{epoch+1}.pth")
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
            input_ids = torch.tensor([dataset.get_input_ids_list(tokenizer, example)], device=accelerator.device)

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
            input_ids = torch.tensor([dataset.get_input_ids_list(tokenizer, example)], device=accelerator.device)

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
        
# -------------------------
# Training loop (memory-friendly)
# -------------------------
def train_conditional_adapter_perhead_cnn(args):
    """
    args must provide:
      args.model_s, args.model_t, args.train_file, args.output_dir,
      args.reuse_a_layer_start (int), args.lr, args.epochs, args.grad_accum_steps,
      args.device (e.g., "cuda:0"), and CNN/model hyperparams (cnn_channels, cond_dim, n_cnn_layers, pred_hidden, dropout)
    """
    device = "cuda:5"
    device_s = "cuda:6"
    print("Device_t:", device)
    print("Device_s:", device_s)

    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load student & teacher (eval, no grad)
    model_s = AutoModelForCausalLM.from_pretrained(args.model_s, trust_remote_code=True).to(device_s)
    model_t = AutoModelForCausalLM.from_pretrained(args.model_t, trust_remote_code=True).to(device)
    model_s.eval(); model_t.eval()
    for p in model_s.parameters(): p.requires_grad = False
    for p in model_t.parameters(): p.requires_grad = False

    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    head_dim_s = getattr(model_s.config, "head_dim", model_s.config.hidden_size // model_s.config.num_attention_heads)
    head_dim_t = getattr(model_t.config, "head_dim", model_t.config.hidden_size // model_t.config.num_attention_heads)
    head_num_s = getattr(model_s.config, "num_key_value_heads", getattr(model_s.config, "num_attention_heads", None))
    head_num_t = getattr(model_t.config, "num_key_value_heads", getattr(model_t.config, "num_attention_heads", None))
    assert head_num_s is not None and head_num_t is not None

    # build per-teacher-layer adapters
    adapters_k = nn.ModuleList([
        ConditionalResidualAdapterPerHeadCNN(
            head_num_t, head_dim_t,
            cnn_channels=getattr(args, "cnn_channels", 256),
            cond_dim=getattr(args, "cond_dim", 128),
            n_cnn_layers=getattr(args, "n_cnn_layers", 3),
            cnn_kernel=getattr(args, "cnn_kernel", 3),
            pred_hidden=getattr(args, "pred_hidden", 256),
            dropout=getattr(args, "dropout", 0.1),
        ) for _ in range(N_t)
    ]).to(device)
    
    adapters_v = nn.ModuleList([
        ConditionalResidualAdapterPerHeadCNN(
            head_num_t, head_dim_t,
            cnn_channels=getattr(args, "cnn_channels", 256),
            cond_dim=getattr(args, "cond_dim", 128),
            n_cnn_layers=getattr(args, "n_cnn_layers", 3),
            cnn_kernel=getattr(args, "cnn_kernel", 3),
            pred_hidden=getattr(args, "pred_hidden", 256),
            dropout=getattr(args, "dropout", 0.1),
        ) for _ in range(N_t)
    ]).to(device)
    
    
    if args.adapter:
        print("Using provided adapter for train.")
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

    optimizer_k = optim.Adam(adapters_k.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer_v = optim.Adam(adapters_v.parameters(), lr=args.lr, weight_decay=0.0)
    mse_loss = nn.MSELoss()

    dataset = HotPotQADataset({"train": args.train_file}, num=10000, split="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    grad_accum_steps = args.grad_accum_steps

    print("Start training per-head conditional adapters (CNN)...")
    for epoch in range(args.epochs):
        adapters_k.train()
        adapters_v.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_loss_k, epoch_loss_v = 0.0, 0.0
        for step, batch in enumerate(pbar):
            example = batch[0]
            input_ids = torch.tensor([dataset.get_input_ids_list(tokenizer, example)], device=device)
            input_ids_s = torch.tensor([dataset.get_input_ids_list(tokenizer, example)], device=device_s)
            # 1) get pkv from teacher & student and move to CPU immediately
            with torch.no_grad():
                s_out = model_s(input_ids=input_ids_s, use_cache=True)
                t_out = model_t(input_ids=input_ids, use_cache=True)
                pkv_s_cpu = [(k.detach().cpu(), v.detach().cpu()) for (k, v) in s_out.past_key_values]
                pkv_t_cpu = [(k.detach().cpu(), v.detach().cpu()) for (k, v) in t_out.past_key_values]
            del s_out, t_out

            if (step % grad_accum_steps) == 0:
                optimizer_k.zero_grad()
                optimizer_v.zero_grad()

            batch_loss_k = 0.0
            batch_loss_v = 0.0
            cnt = 0

            for i in range(args.reuse_a_layer_start, N_t):
                prev_idx = max(0, i - 1)
                s_idx = i

                # pull required pkv from CPU and normalize to [B,H,S,D]
                k_s_prev_cpu, v_s_prev_cpu = pkv_s_cpu[prev_idx]
                k_t_prev_cpu, v_t_prev_cpu = pkv_t_cpu[prev_idx]

                k_s_cur_cpu, v_s_cur_cpu = pkv_s_cpu[s_idx]   # student target
                k_t_cur_cpu, v_t_cur_cpu = pkv_t_cpu[i]            # teacher current source


                # Move only these small tensors to device
                prev_src_k = k_s_prev_cpu.to(device)
                prev_tgt_k = k_t_prev_cpu.to(device)
                cur_src_k = k_s_cur_cpu.to(device)

                prev_src_v = v_s_prev_cpu.to(device)
                prev_tgt_v = v_t_prev_cpu.to(device)
                cur_src_v = v_s_cur_cpu.to(device)

                # For this example we assume no pre-trained base adapter; so base_pred=None
                
                if args.adapter:  
                    base_k_pred = k_mlps(cur_src_k, i)
                    base_v_pred = v_mlps(cur_src_v, i)
                    
                else:
                    k_mlps, v_mlps = None, None
                    base_k_pred, base_v_pred = None, None

                # forward pass
                pred_k = adapters_k[i](
                    prev_src_k=prev_src_k, prev_tgt_k=prev_tgt_k,
                    cur_src_k=cur_src_k, base_k_pred=base_k_pred,
                )
                pred_v = adapters_v[i](
                    prev_src_v, prev_tgt_v,
                    cur_src_v, base_v_pred
                )

                # compute loss against student target (mapped student layer index)
                target_k = k_t_cur_cpu.to(device)
                target_v = v_t_cur_cpu.to(device)

                loss_k = mse_loss(pred_k, target_k) / grad_accum_steps
                loss_v = mse_loss(pred_v, target_v) / grad_accum_steps
                
                batch_loss_k += loss_k.item()
                batch_loss_v += loss_v.item()

                loss_k.backward()
                loss_v.backward()

                cnt += 1

                # free GPU temporaries
                del prev_src_k, prev_tgt_k, cur_src_k
                del prev_src_v, prev_tgt_v, cur_src_v
                del pred_k, pred_v, target_k, target_v, loss_k, loss_v
                torch.cuda.empty_cache()

            if (step + 1) % grad_accum_steps == 0:
                optimizer_k.step()
                optimizer_k.zero_grad()
                optimizer_v.step()
                optimizer_v.zero_grad()

            if cnt > 0:
                avg_loss_k =  batch_loss_k / cnt
                avg_loss_v =  batch_loss_v / cnt
                pbar.set_postfix({"k_loss": avg_loss_k,"v_loss": avg_loss_v})
            else:
                pbar.set_postfix({"avg_loss_per_layer": 0.0})
                
            epoch_loss_k += avg_loss_k
            epoch_loss_v += avg_loss_v

            del pkv_s_cpu, pkv_t_cpu, input_ids
            torch.cuda.empty_cache()

        # save
        save_path = os.path.join(args.output_dir, f"perhead_cond_k_adapters_cnn_epoch{epoch+1}.pth")
        torch.save(adapters_k.state_dict(), save_path)
        print("Saved adapters_k to", save_path)

        save_path = os.path.join(args.output_dir, f"perhead_cond_v_adapters_cnn_epoch{epoch+1}.pth")
        torch.save(adapters_v.state_dict(), save_path)
        print("Saved adapters_v to", save_path)
        
        with open("trainloss.log",'a') as f:
            f.write(f"epoch{epoch}: kloss: {epoch_loss_k / 10000}, vloss: {epoch_loss_v / 10000}")
        
    print("Training finished.")
    
def train_attn_fuser(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_s = AutoModelForCausalLM.from_pretrained(
        args.model_s, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16 ,offload_folder="./tmp/offload_s")
    model_t = AutoModelForCausalLM.from_pretrained(
        args.model_t, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, offload_folder="./tmp/offload_t")
    
    
    model_s.eval().requires_grad_(False)
    model_t.eval().requires_grad_(False)
    
    reuse_a_layer_start = args.reuse_a_layer_start
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    H_s = model_s.config.num_attention_heads
    H_t = model_t.config.num_attention_heads
    
    layer_attn_fuser = LayerAttentionFuser(N_s, N_t, H_s, H_t).to("cpu")
    head_attn_fuser = HeadAttentionFuser(N_s, N_t, H_s, H_t).to("cpu")
    
    ckpt = os.path.join(args.output_dir, f"layer_attn_fuser_epoch{10}.pth")
    state_dict = torch.load(ckpt, map_location="cpu")
    layer_attn_fuser.load_state_dict(state_dict)
    
    ckpt = os.path.join(args.output_dir, f"head_attn_fuser_epoch{10}.pth")
    state_dict = torch.load(ckpt, map_location="cpu")
    head_attn_fuser.load_state_dict(state_dict)
    
    optimizer = AdamW(
        list(layer_attn_fuser.parameters()) + list(head_attn_fuser.parameters()),
        lr=getattr(args, "lr", 1e-4),
        weight_decay=getattr(args, "weight_decay", 0.0)
    )

    # Load eval data
    ds = HotPotQADataset({"train": args.train_file}, begin=1000, end=5000, split="train")
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    print(f"Evaluating on {args.train_file} with {len(ds)} examples.")
    
    for epoch in range(args.epochs):
        layer_attn_fuser.train()
        head_attn_fuser.train()
        total_loss = 0.0
        strlen = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            ex = batch[0]
            input_id_list = ds.get_input_ids_list(tokenizer,ex)
            if step<70:
                strlen = max(strlen, len(input_id_list))
            elif len(input_id_list) > strlen:
                print(f"skip {step}\n")
                continue
            
            device_t = module_device(model_t.get_input_embeddings() or model_t.embed_tokens)
            device_s = module_device(model_s.get_input_embeddings() or model_s.embed_tokens)

            input_ids_s = torch.tensor([input_id_list], device=device_s)
            input_ids_t = torch.tensor([input_id_list], device=device_t)

            # prefill with s
            with torch.inference_mode():
                t_out = model_t(
                    input_ids=input_ids_t,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                
                
            new_t_cache = reuse_attn_distr_with_fuser(model_t, model_s, input_id_list, 
                            layer_attn_fuser, head_attn_fuser, args)
            past = DynamicCache.from_legacy_cache(past_key_values=new_t_cache)
            
            answer = "Answer:" + ex['answer']
            answer_ids = tokenizer(answer).input_ids
            answer_ids = torch.tensor([answer_ids],device=device_t)
            t_out = model_t(
                input_ids=answer_ids,
                use_cache=True,
                past_key_values=past,
                return_dict = True,
                labels = answer_ids
            )
            loss = t_out.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "avg_loss": f"{avg_loss:.6f}"})
            
        os.makedirs(args.output_dir, exist_ok=True)
        save_path_layer = os.path.join(args.output_dir, f"layer_attn_fuser_epoch{epoch+11}.pth")
        save_path_head = os.path.join(args.output_dir, f"head_attn_fuser_epoch{epoch+11}.pth")
        torch.save(layer_attn_fuser.state_dict(), save_path_layer)
        torch.save(head_attn_fuser.state_dict(), save_path_head)
        print(f"✅ Saved adapters to {save_path_layer} and {save_path_head}")
        
from accelerate import Accelerator
from accelerate.utils import DistributedType

def train_attn_fuser_accelerated(args):

    accelerator = Accelerator(
        mixed_precision="fp16",  # 可选 'no', 'fp16', 'bf16'
        gradient_accumulation_steps=getattr(args, "grad_accum_steps", 1),
    )
    
    N_s = len(model_s.model.layers)
    N_t = len(model_t.model.layers)
    H_s = model_s.config.num_attention_heads
    H_t = model_t.config.num_attention_heads


    tokenizer = AutoTokenizer.from_pretrained(args.model_t)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model_s = AutoModelForCausalLM.from_pretrained(
        args.model_s,
        torch_dtype=torch.float16,
        device_map="auto",  # 或 "auto", 但建议 offload 到 CPU
        offload_folder="./tmp/offload_s",
        trust_remote_code=True,
    )
    model_s.eval().requires_grad_(False)

    model_t = AutoModelForCausalLM.from_pretrained(
        args.model_t,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="./tmp/offload_t",
        trust_remote_code=True,
    )
    model_t.eval().requires_grad_(False)


    layer_attn_fuser = LayerAttentionFuser(N_s, N_t, H_s, H_t)
    head_attn_fuser = HeadAttentionFuser(N_s, N_t, H_s, H_t)

    optimizer = AdamW(
        list(layer_attn_fuser.parameters()) + list(head_attn_fuser.parameters()),
        lr=args.lr or 1e-4,
        weight_decay=args.weight_decay or 0.0
    )


    ds = HotPotQADataset({"train": args.train_file}, num=1000, split="train")
    dataloader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda x: x)


    layer_attn_fuser, head_attn_fuser, optimizer, dataloader = accelerator.prepare(
        layer_attn_fuser, head_attn_fuser, optimizer, dataloader
    )

    print(f"Using device: {accelerator.device}")
    print(f"Distributed type: {accelerator.distributed_type}")

    for epoch in range(args.epochs):
        layer_attn_fuser.train()
        head_attn_fuser.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            ex = batch[0]
            input_id_list = ds.get_input_ids_list(tokenizer, ex)

            temperature = args.temperature
            top_p = args.top_p

            with torch.no_grad():
                input_ids_s = torch.tensor([input_id_list]).to(accelerator.device)  # 始终在 CPU
                with accelerator.autocast():
                    s_out = model_s(input_ids=input_ids_s, use_cache=True, output_attentions=False)
                    first_token = sample_next_token(s_out.logits[:, -1, :].squeeze(0), temperature, top_p)
                    last_token_s = torch.tensor([[first_token]]).to(accelerator.device)

                    s_out_next = model_s(input_ids=last_token_s, use_cache=True)
                    logit_label = s_out_next.logits.detach().float()  # [1, vocab]
                    del s_out, s_out_next; torch.cuda.empty_cache()

                # ==== Teacher 推理（也尽量 offload）====
                input_ids_t = torch.tensor([input_id_list]).to(accelerator.device)
                t_out = model_t(input_ids=input_ids_t, use_cache=True)
                first_token = sample_next_token(t_out.logits[:, -1, :].squeeze(0), temperature, top_p)
                last_token_t = torch.tensor([[first_token]]).to(accelerator.device)
                pkv_t_cpu = t_out.past_key_values
                del t_out; torch.cuda.empty_cache()

            # ==== 使用 fuser 构造新的 KV cache ====
            # 这个函数内部要小心处理设备转移
            new_pkv = reuse_attn_distr_with_fuser_accelerated(
                model_t, model_s, input_id_list,
                layer_attn_fuser, head_attn_fuser, args,
                pkv_t_cpu  # 已在 CPU
            )
            past = DynamicCache.from_legacy_cache(new_pkv)

            # ==== 最后一步：只把需要的部分移到 GPU 训练 fuser 输出 ====
            last_token_t = last_token_t.to(accelerator.device)
            with accelerator.autocast():
                out = model_t(
                    input_ids=last_token_t,
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=False,
                )
                logits = out.logits
                logit_label = logit_label.to(logits.device)

                loss = distillation_loss(logits, logit_label, temperature=temperature)

            # 反向传播由 accelerator 管理
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.6f}", "avg_loss": f"{avg_loss:.6f}"})

        # ==== 保存模型 ====
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path_layer = os.path.join(args.output_dir, f"layer_attn_fuser_epoch{epoch+1}.pth")
            save_path_head = os.path.join(args.output_dir, f"head_attn_fuser_epoch{epoch+1}.pth")

            # Save only current process weights
            unwrapped_layer = accelerator.unwrap_model(layer_attn_fuser)
            unwrapped_head = accelerator.unwrap_model(head_attn_fuser)

            torch.save(unwrapped_layer.state_dict(), save_path_layer)
            torch.save(unwrapped_head.state_dict(), save_path_head)
            print(f"✅ Saved adapters to {save_path_layer} and {save_path_head}")
