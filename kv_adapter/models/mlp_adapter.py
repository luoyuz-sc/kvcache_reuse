# ---------------- MLP Adapter ----------------
import torch
import torch.nn as nn
from typing import Optional
from types import List

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
class KAdapter(nn.Module):
    """
    Adapter that converts source K (multiple heads) -> target-expected K shape.
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

        self.key_embed = nn.Linear(send_heads * send_head_dim, hidden_dim)
        self.key_mlp = RegularMLP(hidden_dim, intermediate_dim, num_layers)
        self.key_out = nn.Linear(hidden_dim, receive_heads * receive_head_dim)
        self.key_norm = RMSNorm(receive_head_dim)

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

        # 1)[B, H, S, D] -> [B, S, H*D]
        x_comb = x.permute(0,2,1,3).reshape(B, S, Sh * D)

        # 2) key path
        k = self.key_embed(x_comb)            # [B, S, hidden_dim]
        k_flat = k.reshape(B * S, self.hidden_dim)
        k_hidden = self.key_mlp(k_flat)       # [N, hidden_dim]
        k_out_flat = self.key_out(k_hidden)   # [B*S, receive_heads * receive_head_dim]
        k_out = k_out.reshape(B, S, self.receive_heads, self.receive_head_dim)
        k_out = self.key_norm(k_out)  # RMSNorm 在最后一维归一化
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