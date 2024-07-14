from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ModelArgs:
    def __init__(
        self,
        dim: int,
        n_layers: int,
        head_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        vocab_size: int,
        rope_theta: float = 10000,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.rope_theta = rope_theta


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def _norm(self, x):
        return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt()

    def forward(self, x):
        output = self._norm(x.float()).type(x.dtype)
        return self.weight * output


class FeedForward(nn.Module):
    def __init__(self, args: "ModelArgs"):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    def __init__(self, dim: int, traditional: bool = False, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.traditional = traditional
        self.base = base
        self.freqs = self.create_freqs(dim // 2)

    def create_freqs(self, n: int):
        freqs = 1.0 / (self.base ** (torch.arange(0, n, 2) / n))
        return freqs

    def forward(self, x: torch.Tensor, offset: int = 0):
        if self.traditional:
            t = torch.arange(x.shape[2], device=x.device) + offset
        else:
            t = torch.arange(x.shape[2], device=x.device)
        freqs = self.freqs.to(x.device)
        t_sin = torch.sin(t[:, None] * freqs[None, :])
        t_cos = torch.cos(t[:, None] * freqs[None, :])
        return torch.stack(
            [
                x[..., 0::2] * t_cos + x[..., 1::2] * t_sin,
                x[..., 0::2] * t_sin - x[..., 1::2] * t_cos,
            ],
            dim=-1,
        ).flatten(-2, -1)


class Attention(nn.Module):
    def __init__(self, args: "ModelArgs"):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = RoPE(args.head_dim, traditional=True, base=args.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.view(B, L, self.n_heads, -1).transpose(1, 2)
        keys = keys.view(B, L, self.n_kv_heads, -1).transpose(1, 2)
        values = values.view(B, L, self.n_kv_heads, -1).transpose(1, 2)

        def repeat(a):
            a = torch.cat([a.unsqueeze(2)] * self.repeats, dim=2)
            return a.view([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        # Rolling Buffer Cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = torch.cat([key_cache, keys], dim=2)
            values = torch.cat([value_cache, values], dim=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(-1, -2)
        if mask is not None:
            scores += mask
        scores = F.softmax(scores.float(), dim=-1).type(scores.dtype)
        output = (scores @ values).transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(output), (keys, values)


class TransformerBlock(nn.Module):
    def __init__(self, args: "ModelArgs"):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Mistral(nn.Module):
    def __init__(self, args: "ModelArgs"):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = self._generate_square_subsequent_mask(h.shape[1]).to(h.device)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), cache
