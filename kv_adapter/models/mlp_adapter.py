# ---------------- MLP Adapter ----------------
import torch
import torch.nn as nn
from typing import Optional,List
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

        self.key_embed = nn.Linear(send_heads * send_head_dim, hidden_dim)

        self.key_mlp = RegularMLP(hidden_dim, intermediate_dim, num_layers)

        self.key_out = nn.Linear(hidden_dim, receive_heads * receive_head_dim)

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

        #  [B, S, send_heads * send_head_dim]
        x_comb = x.permute(0,2,1,3).reshape(B, S, Sh * D)

        #  key path
        k = self.key_embed(x_comb)            # [B, S, hidden_dim]
        k_flat = k.reshape(B * S, self.hidden_dim)
        k_hidden = self.key_mlp(k_flat)       # [N, hidden_dim] -> [N, hidden_dim]
        k_out_flat = self.key_out(k_hidden)   # [B*S, receive_heads * receive_head_dim]
        k_out_flat = self.key_norm(k_out_flat)  # RMSNorm 
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
    
import torch
import torch.nn as nn
from typing import List, Union, Optional

class RidgeAdapter(nn.Module):
    """
    Ridge regression per head (closed-form).
    - fit(X, Y): X can be list of n tensors shape [d] or a tensor [n, d].
                 Y same shape.
    - predict(x): x shape [d] or [m, d] -> returns [d] or [m, d]
    Stores W (d,d) and b (d,) as buffers (not learnable params).
    """
    def __init__(self, alpha: float = 1e-2, fit_intercept: bool = True, device: str = 'cpu'):
        super().__init__()
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        # buffers to be set after fit; initially None
        self.register_buffer('_W', None)  # will hold [d, d]
        self.register_buffer('_b', None)  # will hold [d]
        self._device_for_fit = torch.device(device)

    @property
    def W(self) -> Optional[torch.Tensor]:
        return self._W

    @property
    def b(self) -> Optional[torch.Tensor]:
        return self._b

    def _to_matrix(self, X: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Convert input to shape [n, d] (n samples).
        Accepts:
          - list of n tensors each shape [d]
          - tensor [n, d]
        """
        if isinstance(X, list):
            if len(X) == 0:
                return torch.empty((0, 0))
            # stack on first dim -> [n, d]
            Xt = torch.stack([x.detach().cpu() for x in X], dim=0)
            return Xt
        if not torch.is_tensor(X):
            raise TypeError("X must be list[tensor] or tensor")
        if X.dim() == 1:
            return X.unsqueeze(0)
        if X.dim() == 2:
            return X.detach().cpu()
        raise ValueError(f"Unsupported tensor shape for X: {X.shape}")

    @torch.no_grad()
    def fit(self, X: Union[List[torch.Tensor], torch.Tensor],
                  Y: Union[List[torch.Tensor], torch.Tensor],
                  alpha: Optional[float] = None,
                  device_for_storage: Optional[Union[str, torch.device]] = None):
        """
        Fit ridge regression: Y = W X + b  (all column vectors)
        Inputs:
          X: list of n tensors [d] or tensor [n, d]
          Y: same shape as X
          alpha: regularization (overrides self.alpha if provided)
          device_for_storage: where to store W and b (defaults to module device)
        After fit, W and b are registered buffers and can be moved with .to().
        """
        if alpha is None:
            alpha = self.alpha
        else:
            alpha = float(alpha)

        X_mat = self._to_matrix(X)  # [n, d]
        Y_mat = self._to_matrix(Y)  # [n, d]

        if X_mat.numel() == 0:
            raise ValueError("No samples provided to fit()")

        if X_mat.shape != Y_mat.shape:
            raise ValueError(f"X and Y must have same shape: got {X_mat.shape} vs {Y_mat.shape}")

        # input dims
        n, d = X_mat.shape

        # Convert to float64 for better numeric stability during linear algebra, then back later
        dtype = X_mat.dtype
        Xc = X_mat.to(torch.float64)
        Yc = Y_mat.to(torch.float64)

        # center
        x_mean = Xc.mean(dim=0, keepdim=True)  # [1, d]
        y_mean = Yc.mean(dim=0, keepdim=True)

        Xc = (Xc - x_mean).T  # now [d, n]
        Yc = (Yc - y_mean).T  # [d, n]

        # compute A = Xc Xc^T + alpha I  (d x d)
        A = Xc @ Xc.T  # [d, d]
        if alpha > 0:
            A = A + alpha * torch.eye(d, dtype=A.dtype)

        # compute RHS = Xc Yc^T  => shape [d, d]  (note earlier derivation)
        # We want W such that Yc = W Xc -> W = Yc Xc^T (Xc Xc^T + αI)^{-1}
        # Use linear solve for stability: solve A^T * W^T = (Xc @ Yc.T)
        RHS = Xc @ Yc.T  # [d, d]

        # Solve for W^T: A @ W^T = RHS  -> W^T = solve(A, RHS)
        # use torch.linalg.solve which expects square matrix A and RHS
        # convert to float64 for solve
        Wt = torch.linalg.solve(A, RHS)  # [d, d]
        W = Wt.T  # [d, d]  (float64)

        # compute bias
        if self.fit_intercept:
            b = (y_mean.T - W @ x_mean.T).squeeze(1)  # [d]
        else:
            b = torch.zeros((d,), dtype=W.dtype)

        # cast back to original dtype for storage
        W = W.to(dtype)
        b = b.to(dtype)

        # store as buffers (on cpu) then optionally move to requested device
        # register_buffer was used so we replace by setattr
        # we need to store on cpu to avoid accidental GPU fill during fit
        device_for_storage = torch.device(device_for_storage) if device_for_storage is not None else next(self.parameters(), None)
        # if next(self.parameters()) is None use cpu
        if device_for_storage is None:
            device_for_storage = torch.device('cpu')
        elif isinstance(device_for_storage, torch.nn.parameter.Parameter):
            device_for_storage = device_for_storage.device

        # move to desired storage device
        W = W.to(device_for_storage)
        b = b.to(device_for_storage)

        # assign buffers (can't re-register, but assign via setattr)
        # Ensure consistent dtype/device with module
        object.__setattr__(self, '_W', W)
        object.__setattr__(self, '_b', b)

        return self

    def predict(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Predict y = W @ x + b
        x: tensor [d] or [m, d], or list of vectors -> stacked to [m, d]
        returns tensor with same leading shape: [d] or [m, d]
        """
        if self._W is None or self._b is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        # Stack input
        if isinstance(x, list):
            X = torch.stack([xi.detach() for xi in x], dim=0)  # [m, d]
        elif torch.is_tensor(x):
            if x.dim() == 1:
                X = x.unsqueeze(0)
            elif x.dim() == 2:
                X = x
            else:
                raise ValueError("x must be 1D or 2D tensor")
        else:
            raise TypeError("x must be tensor or list of tensors")

        # ensure dtype/device match
        W = self._W
        b = self._b
        X = X.to(W.dtype).to(W.device)

        # X shape [m, d], we want y = X @ W.T + b
        y = X @ W.T  # [m, d]
        y = y + b.unsqueeze(0)

        # if input was 1D, return 1D
        if y.shape[0] == 1:
            return y.squeeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias to predict to fit nn.Module pattern."""
        assert x.dim() == 4, "expected input shape [B, S, send_heads, send_head_dim]"
        B, Sh, S, D = x.shape
        #  [B, S, send_heads * send_head_dim]
        x = x.permute(0,2,1,3).reshape(B*S, Sh * D)
        x = self.predict(x)
        x = x.reshape(B,S,Sh,D)
        x = x.permute(0,2,1,3)
        return x


# -------------------------
# Model: per-head conditional CNN adapter
# -------------------------
class ConditionalResidualAdapterPerHeadCNN(nn.Module):
    """
    Per-head conditional residual adapter (CNN-based encoder).
    - prev_src_k, prev_tgt_k: [B, H, S, D]
    - cur_src_k: [B, H, S, D]
    - base_* optional: [B, H, S, D]
    Output:
    - pred_k: [B, H, S, D]
    """
    def __init__(
        self,
        H: int,
        D: int,
        cnn_channels: int = 256,
        cond_dim: int = 128,
        n_cnn_layers: int = 3,
        cnn_kernel: int = 3,
        pred_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # meta
        self.H_prev = H
        self.D_prev = D
        self.H_cur = H
        self.D_cur = D
        self.D_out = D
        self.cond_dim = cond_dim
        self.cnn_channels = cnn_channels

        # per-head per-token feature dim: concat(src_head_token, tgt_head_token, residual) -> 3 * D_prev
        self.per_head_feat = 3 * D

        # Input projection per head: maps 3*D_prev -> cnn_channels
        # We'll apply this per head in a batched manner by stacking heads into batch dimension.
        self.head_input_proj = nn.Linear(self.per_head_feat, cnn_channels)

        # CNN stack operating over sequence dimension; we will apply it per-head (batched)
        cnn_layers = []
        in_ch = cnn_channels
        for _ in range(n_cnn_layers):
            cnn_layers.append(nn.Conv1d(in_ch, in_ch, kernel_size=cnn_kernel, padding=cnn_kernel//2))
            cnn_layers.append(nn.GELU())
            # layernorm expects [B, S, C], so we will permute before applying it; implement as nn.GroupNorm(1,C) which works on [B,C,S]
            cnn_layers.append(nn.GroupNorm(1, in_ch))
        self.cnn = nn.Sequential(*cnn_layers)

        # project cnn output -> cond_dim per head per token
        self.head_cond_proj = nn.Linear(cnn_channels, cond_dim)

        # mapper: map concatenated per-prev-head conds -> per-current-head conds
        # input per token shape: [H_prev * cond_dim] -> output [H_cur * cond_dim]
        self.cond_mapper = nn.Linear(H * cond_dim, H * cond_dim)

        # predictor: per head per token MLP: input D_cur + cond_dim -> D_out
        pred_input = D + cond_dim
        self.pred_k = nn.Sequential(
            nn.Linear(pred_input, pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, D)
        )

    def forward(self, prev_src_k, prev_tgt_k, cur_src_k, base_k_pred=None):
        """
        Shapes:
         - prev_src_k, prev_tgt_k: [B, H_prev, S, D_prev]
         - cur_src_k: [B, H_cur, S, D_cur]
        Returns:
         - pred_k, pred_v: [B, H_cur, S, D_out]
        """
        B, H_prev, S, D_prev = prev_src_k.shape
        assert H_prev == self.H_prev and D_prev == self.D_prev
        B2, Hc, S2, Dc = cur_src_k.shape
        assert B == B2 and S == S2 and Hc == self.H_cur and Dc == self.D_cur

        device = cur_src_k.device

        # ---- per-head per-token features from prev layer ----
        # prev_src_k: [B, H_prev, S, D_prev] -> permute -> [B, S, H_prev, D_prev] -> reshape to [B*S*H_prev, D_prev]
        prev_src_perm = prev_src_k.permute(0, 2, 1, 3).contiguous().view(B * S * H_prev, D_prev)
        prev_tgt_perm = prev_tgt_k.permute(0, 2, 1, 3).contiguous().view(B * S * H_prev, D_prev)
        residual_perm = (prev_tgt_perm - prev_src_perm)  # [B*S*H_prev, D_prev]

        # concat per-head features: [B*S*H_prev, 3*D_prev]
        head_feats = torch.cat([prev_src_perm, prev_tgt_perm, residual_perm], dim=-1)  # [B*S*H_prev, 3*D_prev]

        # project per-head feature -> cnn_channels
        head_proj = self.head_input_proj(head_feats)  # [B*S*H_prev, cnn_channels]

        # reshape to per-head "batch" for Conv1d: we need shape [B*H_prev, C, S]
        head_proj_seq = head_proj.view(B * H_prev, S, self.cnn_channels).permute(0, 2, 1).contiguous()  # [B*H_prev, C, S]

        # apply CNN stack (shared across heads)
        cnn_out = self.cnn(head_proj_seq)  # [B*H_prev, C, S]

        # back to [B*H_prev, S, C]
        cnn_out = cnn_out.permute(0, 2, 1).contiguous()  # [B*H_prev, S, C]

        # project to per-head per-token cond: [B*H_prev, S, cond_dim]
        head_cond = self.head_cond_proj(cnn_out)  # [B*H_prev, S, cond_dim]

        # reshape to [B, H_prev, S, cond_dim]
        head_cond = head_cond.view(B, H_prev, S, self.cond_dim)

        # ---- map per-prev-head conds to per-cur-head conds ----
        # first reshape per token: [B, S, H_prev * cond_dim]
        per_token_prev = head_cond.permute(0, 2, 1, 3).contiguous().view(B, S, H_prev * self.cond_dim)  # [B,S,H_prev*cond_dim]

        # map via linear to [B, S, H_cur*cond_dim]
        mapped = self.cond_mapper(per_token_prev)  # [B, S, H_cur*cond_dim]

        # reshape to [B, S, H_cur, cond_dim] -> permute to [B, H_cur, S, cond_dim]
        mapped = mapped.view(B, S, Hc, self.cond_dim).permute(0, 2, 1, 3).contiguous()  # [B,H_cur,S,cond_dim]

        cond_cur = mapped  # [B, H_cur, S, cond_dim]

        # ---- concat cond_cur with cur_src_k per head token and predict residual ----
        # cur_src_k: [B, H_cur, S, D_cur]
        cur_concat = torch.cat([cur_src_k, cond_cur], dim=-1)  # [B,H_cur,S, D_cur+cond_dim]
        feat = cur_concat.view(B * Hc * S, -1)  # [B*Hc*S, D_cur+cond_dim]

        delta_k = self.pred_k(feat).view(B, Hc, S, self.D_out)     
        # add base predictions if provided
        if base_k_pred is None:
            pred_k = delta_k
        else:
            pred_k = base_k_pred + delta_k

        return pred_k