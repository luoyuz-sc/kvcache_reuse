import gc, torch, inspect, types

def debug_report_gpu_tensors(device):
    device = torch.device(device)
    objs = gc.get_objects()
    tensor_objs = []
    for o in objs:
        try:
            # only torch.Tensor or objects containing them
            if torch.is_tensor(o):
                if o.device == device:
                    tensor_objs.append((type(o).__name__, o.shape, o.element_size()*o.numel()))
            elif hasattr(o, "__dict__"):
                # minor attempt to find tensors inside objects
                for v in vars(o).values():
                    if torch.is_tensor(v) and v.device == device:
                        tensor_objs.append((type(o).__name__, v.shape, v.element_size()*v.numel()))
        except Exception:
            continue
    # aggregate by (type,shape) to reduce noise
    from collections import Counter
    ctr = Counter((typ, tuple(shape), size) for typ,shape,size in tensor_objs)
    print(">>> GPU tensor summary (device={}): total objects {}, total unique kinds {}".format(
        device, len(tensor_objs), len(ctr)))
    for (typ, shape, size), count in ctr.most_common(20):
        print(f"{count:4d} x {typ:20s} shape={shape} bytes_each={size:,}")
    # memory snapshot
    print("torch.cuda.memory_allocated:", torch.cuda.memory_allocated(device))
    print("torch.cuda.memory_reserved :", torch.cuda.memory_reserved(device))