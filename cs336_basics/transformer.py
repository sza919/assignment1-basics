from __future__ import annotations
from einops import einsum, rearrange
import torch
import numpy as np
from jaxtyping import Float, Int

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = np.sqrt(2/(in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, std = std, a = -3*std, b = 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... input, output input -> ... output')
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        ) # not a linear mapping, just need a matrix to store the embeddings
        std = 1
        torch.nn.init.trunc_normal_(self.weight, std = std, a = -3*std, b = 3*std)

    # The forward method should select the embedding vector for each token ID by 
    # indexing into an embedding matrix of shape (vocab_size, d_model) using a torch.LongTensor 
    # of token IDs with shape (batch_size, sequence_length).
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )
        norm_x = x / rms
        return (norm_x * self.weight).to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(self, dmodel:int, dff:int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(dmodel, dff, device=device, dtype=dtype)
        self.w3 = Linear(dmodel, dff, device=device, dtype=dtype)
        self.w2 = Linear(dff, dmodel, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1x = einsum(x, self.w1.weight, '... dmodel, dff dmodel -> ... dff')
        W3x = einsum(x, self.w3.weight, '... dmodel, dff dmodel -> ... dff')
        return einsum(torch.sigmoid(W1x) * W1x * W3x, self.w2.weight, '... dff, dmodel dff -> ... dmodel')
    
class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None  = None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        thetas = einsum(torch.arange(max_seq_len, device=device).float(), inv_freq, 'seq_len, d_half -> seq_len d_half')
        sinusoids = torch.sin(thetas)
        cosinoids = torch.cos(thetas)
        self.register_buffer('sinusoids', sinusoids, persistent=False)
        self.register_buffer('cosinoids', cosinoids, persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        sinusoids = self.sinusoids[token_positions] #'... seq_len d_model/2'
        cosinoids = self.cosinoids[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x1_rotated = x1 * cosinoids - x2 * sinusoids
        x2_rotated = x1 * sinusoids + x2 * cosinoids
        out = torch.empty_like(x)
        out[..., 0::2] = x1_rotated
        out[..., 1::2] = x2_rotated
        return out

def softmax(x: torch.Tensor, dim: int):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def ScaledDotProductAttention(
        keys: Float[Tensor, " ... seq_len d_k"],
        queries: Float[Tensor, " ... seq_len d_k"],
        values: Float[Tensor, " ... seq_len d_v"],
        mask: Float[Tensor, "seq_len seq_len"] | None = None
) -> Float[Tensor, " ... seq_len_q d_v"]:
    scores = einsum(queries, 
                    keys, 
                    '... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k'
                    ) / np.sqrt(keys.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention = einsum(softmax(scores, dim = -1),
                        values,
                          '... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v')
    return attention

class MultiHeadAttention(torch.nn.Module):
    def __init__(
            self,
            d_model: int, 
            num_heads: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        rope = None
    ) -> Float[Tensor, " ... seq_len_q d_model"]:
        Q = einsum(x, self.q_proj.weight, '... seq_len d_model, dq d_model -> ... seq_len dq')
        K = einsum(x, self.k_proj.weight, '... seq_len d_model, dk d_model -> ... seq_len dk')
        V = einsum(x, self.v_proj.weight, '... seq_len d_model, dv d_model -> ... seq_len dv')
        
        Q_heads = rearrange(Q, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        K_heads = rearrange(K, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        V_heads = rearrange(V, '... seq_len (num_heads d_v) -> ... num_heads seq_len d_v', num_heads=self.num_heads)
        if rope is not None:
            seq_len = Q_heads.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.expand(*Q_heads.shape[:-2], seq_len)
            Q_heads = rope.forward(Q_heads, token_positions)
            K_heads = rope.forward(K_heads, token_positions)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device)).bool()
        heads = ScaledDotProductAttention(K_heads, Q_heads, V_heads, mask)
        heads_concat = rearrange(heads, '... num_heads seq_len d_v -> ... seq_len (num_heads d_v)')
        return einsum(heads_concat, self.output_proj.weight, '... seq_len h_d_v, d_model h_d_v -> ... seq_len d_model')

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta, max_seq_len, device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model, device = device, dtype = dtype)
        self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)
        self.rope = RoPE(theta, d_model//num_heads, max_seq_len, device = device)
        self.attn = MultiHeadAttention(d_model, num_heads, device = device, dtype = dtype)
        self.ffn = SwiGLU(d_model, d_ff, device = device, dtype = dtype)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        y = x + self.attn(self.ln1(x), self.rope)
        output = y + self.ffn(self.ln2(y))
        return output
    
class TransformerLM(torch.nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 theta: float, 
                 vocab_size: int, 
                 context_length: int, 
                 num_layers: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device = device, dtype = dtype)

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:

        h = self.token_embeddings(x)

        for layer in self.layers:
            h = layer(h)

        h = self.ln_final(h)
        h = self.lm_head(h)
        return h
