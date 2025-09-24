#!/usr/bin/env python3
"""Pure-MLX implementation of the quantized Qwen3 Next 80B A3B model.

This script rebuilds the Qwen3 Next architecture directly with MLX layers,
loads the 4-bit quantized weights published at
``mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`` and offers a minimal CLI for
prompting the model.  It avoids the higher-level ``mlx_lm.load`` helper so that
all of the model wiring, quantization conversion, and sampling loop are easy to
inspect and customize.

Usage example (weights must already be downloaded or will be fetched on first
run)::

    python mlx_qwen3_next.py --prompt "Hello" --max-new-tokens 64 --stream

Because the 80B checkpoint still occupies roughly 40 GB even in 4-bit form, the
script is primarily intended for Apple Silicon hosts with ample unified memory
(e.g. M3 Ultra with 96 GB).  Generation defaults to nucleus sampling and works
without past-key-value caches for simplicity, but the architecture mirrors the
official DeltaNet and attention blocks so it can be extended easily.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

try:  # Optional at import time so docs/tests can run without MLX present.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.nn.layers.quantized import quantize as quantize_layers
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover - exercised only on non-MLX hosts.
    mx = None  # type: ignore
    nn = None  # type: ignore
    quantize_layers = None  # type: ignore
    tree_flatten = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DEFAULT_MODEL_ID = "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"


def require_mlx() -> None:
    """Ensure MLX is available before using any tensor-heavy functionality."""

    if mx is None or nn is None or quantize_layers is None or tree_flatten is None:
        raise RuntimeError(
            "MLX is required for this script. Install the `mlx` wheel on macOS "
            "and ensure it is importable before running generation."
        ) from _IMPORT_ERROR


@dataclass
class Qwen3NextConfig:
    """Minimal configuration container mirroring the Hugging Face JSON fields."""

    model_type: str = "qwen3_next"
    hidden_size: int = 4096
    num_hidden_layers: int = 28
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 8
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    mlp_only_layers: List[int] = field(default_factory=list)
    moe_intermediate_size: int = 0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 0
    num_key_value_heads: int = 0
    rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.5
    max_position_embeddings: int = 65536
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    full_attention_interval: int = 4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Qwen3NextConfig":
        params: Dict[str, Any] = {}
        for name, field_info in cls.__dataclass_fields__.items():  # type: ignore
            if name in data:
                params[name] = data[name]
            elif field_info.default is not MISSING:
                params[name] = field_info.default
            elif field_info.default_factory is not MISSING:  # type: ignore
                params[name] = field_info.default_factory()  # type: ignore
        if not params.get("mlp_only_layers"):
            params["mlp_only_layers"] = data.get("mlp_only_layers", [])
        if params.get("shared_expert_intermediate_size", 0) == 0:
            params["shared_expert_intermediate_size"] = data.get(
                "shared_expert_intermediate_size", params["intermediate_size"]
            )
        if params.get("moe_intermediate_size", 0) == 0:
            params["moe_intermediate_size"] = data.get(
                "moe_intermediate_size", params["intermediate_size"]
            )
        if params.get("num_key_value_heads", 0) == 0:
            params["num_key_value_heads"] = data.get(
                "num_key_value_heads", params["linear_num_key_heads"]
            )
        head_dim = params.get("head_dim")
        if head_dim in (None, 0):
            params["head_dim"] = params["hidden_size"] // params["num_attention_heads"]
        return cls(**params)


# ---------------------------------------------------------------------------
# Rotary embeddings, normalization helpers, and gated DeltaNet utilities
# ---------------------------------------------------------------------------


def zero_centered_rms_norm(
    hidden_states: mx.array, weight: Optional[mx.array], eps: float
) -> mx.array:
    """RMS normalization that also subtracts the mean along the last dimension."""

    centered = hidden_states - hidden_states.mean(axis=-1, keepdims=True)
    inv_rms = mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + eps)
    out = centered * inv_rms
    if weight is not None:
        out = out * weight
    return out


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: Optional[Dict[str, Any]] = None,
    max_position_embeddings: Optional[int] = None,
) -> nn.Module:
    """Instantiate a rotary positional embedding module."""

    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"

    if rope_type in {"default", "linear"}:
        scale = 1.0
        if rope_type == "linear":
            scale = 1 / scaling_config["factor"]  # type: ignore[index]
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    if rope_type == "llama3":
        from math import ceil

        factor = scaling_config["factor"]  # type: ignore[index]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get("original_max_position_embeddings", 8192)
        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * math.pi * freqs
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        scaled = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth = (
            old_context_len / wavelens - low_freq_factor
        ) / (high_freq_factor - low_freq_factor)
        smooth_freqs = freqs / ((1 - smooth) / factor + smooth)
        final_freqs = mx.where(is_medium, smooth_freqs, scaled)

        class Llama3RoPE(nn.Module):
            def __init__(self, dims: int):
                super().__init__()
                self.dims = dims

            def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
                return mx.fast.rope(
                    x,
                    self.dims,
                    traditional=traditional,
                    base=None,
                    scale=1.0,
                    offset=offset,
                    freqs=final_freqs,
                )

        return Llama3RoPE(dims)

    if rope_type == "yarn":
        scaling_factor = scaling_config["factor"]  # type: ignore[index]
        orig = scaling_config.get("original_max_position_embeddings", 4096)
        beta_fast = scaling_config.get("beta_fast", 32)
        beta_slow = scaling_config.get("beta_slow", 1)
        mscale = scaling_config.get("mscale", 1)
        mscale_all_dim = scaling_config.get("mscale_all_dim", 0)

        def yarn_find_correction_dim(num_rotations: float) -> float:
            return (
                dims
                * math.log(orig / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = math.floor(yarn_find_correction_dim(beta_fast))
        high = math.ceil(yarn_find_correction_dim(beta_slow))
        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freqs
        freq_extra = freqs
        if low == high:
            high += 1
        ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
        ramp = mx.clip(ramp, 0, 1)
        blended = (freq_inter * freq_extra) / (freq_inter * ramp + freq_extra * (1 - ramp))
        mscale_factor = 1.0
        if scaling_factor > 1:
            mscale_factor = 0.1 * mscale * math.log(scaling_factor) + 1.0
            mscale_factor /= 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0

        class YarnRoPE(nn.Module):
            def __init__(self, dims: int):
                super().__init__()
                self.dims = dims

            def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
                if mscale_factor != 1.0:
                    x[..., : self.dims] = mscale_factor * x[..., : self.dims]
                return mx.fast.rope(
                    x,
                    self.dims,
                    traditional=traditional,
                    base=None,
                    scale=1.0,
                    offset=offset,
                    freqs=blended,
                )

        return YarnRoPE(dims)

    if rope_type == "longrope":
        short_factor = scaling_config["short_factor"]  # type: ignore[index]
        long_factor = scaling_config["long_factor"]  # type: ignore[index]
        orig = scaling_config["original_max_position_embeddings"]  # type: ignore[index]
        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freqs = mx.array(long_factor, dtype=mx.float32) * freqs
        scale = math.sqrt(1 + math.log(max_position_embeddings / orig) / math.log(orig))

        class LongRoPE(nn.Module):
            def __init__(self, dims: int):
                super().__init__()
                self.dims = dims

            def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
                x[..., : self.dims] = scale * x[..., : self.dims]
                return mx.fast.rope(
                    x,
                    self.dims,
                    traditional=False,
                    base=None,
                    scale=1.0,
                    offset=offset,
                    freqs=freqs,
                )

        return LongRoPE(dims)

    return nn.RoPE(dims, traditional=traditional, base=base)


from functools import partial


@partial(mx.compile, shapeless=True)
def _compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias).astype(A_log.dtype))


def _gated_delta_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    state = state * g[..., None, None]
    kv_mem = (state * k[..., None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None, :] * delta[..., None]
    y = (state * q[..., None, :]).sum(axis=-1)
    return y, state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    beta = mx.sigmoid(b)
    g = _compute_g(A_log, a, dt_bias)
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)
    if (repeat := Hv // Hk) > 1:
        q = mx.repeat(q, repeat, -2)
        k = mx.repeat(k, repeat, -2)
    outputs = []
    for t in range(T):
        y, state = _gated_delta_step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state)
        outputs.append(y)
    return mx.stack(outputs, axis=1), state


# ---------------------------------------------------------------------------
# Switch-GLU helpers (MOE layers). Included for completeness although the
# 80B A3B checkpoint does not enable the sparse experts.
# ---------------------------------------------------------------------------


def _gather_sort(x: mx.array, indices: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    *_, M = indices.shape
    indices_flat = indices.flatten()
    order = mx.argsort(indices_flat)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices_flat[order], inv_order


def _scatter_unsort(x: mx.array, inv_order: mx.array, shape: Optional[Tuple[int, ...]] = None) -> mx.array:
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return nn.silu(gate) * x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        self.freeze()

    @property
    def input_dims(self) -> int:
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False) -> mx.array:
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self) -> int:
        return self.weight.shape[2]

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False) -> mx.array:
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        ql = QuantizedSwitchLinear(
            self.input_dims,
            self.output_dims,
            self.num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight,
            group_size,
            bits,
            mode=mode,
        )
        ql.biases = biases[0] if biases else None
        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: Optional[nn.Module] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.activation = activation or SwiGLU()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        else:
            x = mx.unflatten(x, 0, indices.shape)
        return x.sum(axis=-3)


# ---------------------------------------------------------------------------
# Core Qwen3 Next architecture in MLX
# ---------------------------------------------------------------------------


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return zero_centered_rms_norm(hidden_states, self.weight, self.eps)


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = ZeroCenteredRMSNorm(hidden_size, eps)

    def __call__(self, hidden_states: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        x = self.norm(hidden_states)
        if gate is not None:
            x = x * nn.silu(gate)
        return x


class Qwen3NextAttention(nn.Module):
    def __init__(self, args: Qwen3NextConfig):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        q_proj = self.q_proj(x)
        queries, gate = mx.split(
            q_proj.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        attn = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn * mx.sigmoid(gate))


class Qwen3NextMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "Number of value heads must be divisible by number of key heads"
            )
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        self.in_proj_qkvz = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False
        )
        self.in_proj_ba = nn.Linear(
            self.hidden_size, self.num_v_heads * 2, bias=False
        )
        self.dt_bias = mx.ones(self.num_v_heads)
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_ordering(
        self, mixed_qkvz: mx.array, mixed_ba: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        nk, dv, nv = self.num_k_heads, self.head_v_dim, self.num_v_heads
        mixed_qkvz = mixed_qkvz.reshape(*mixed_qkvz.shape[:-1], nk, -1)
        mixed_ba = mixed_ba.reshape(*mixed_ba.shape[:-1], nk, -1)
        q, k, v, z = mx.split(mixed_qkvz, [self.key_dim, 2 * self.key_dim, 2 * self.key_dim + nv // nk * dv], axis=-1)
        b, a = mx.split(mixed_ba, [nv // nk], axis=-1)
        return (
            q,
            k,
            v.reshape(*v.shape[:2], -1, dv),
            z.reshape(*z.shape[:2], -1, dv),
            b.reshape(*b.shape[:2], nv),
            a.reshape(*a.shape[:2], nv),
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        B, S, _ = inputs.shape
        q, k, v, z, b, a = self.fix_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        )
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
        )
        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * zero_centered_rms_norm(q, None, 1e-6)
        k = inv_scale * zero_centered_rms_norm(k, None, 1e-6)
        out, _ = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args: Qwen3NextConfig):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_size = args.shared_expert_intermediate_size
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, self.num_experts)
        self.shared_expert = Qwen3NextMLP(dim, shared_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        indices = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        expert_out = self.switch_mlp(x, indices)
        expert_out = (expert_out * scores[..., None]).sum(axis=-2)
        shared = self.shared_expert(x)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
        return expert_out + shared


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, args: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = Qwen3NextGatedDeltaNet(args)
        else:
            self.self_attn = Qwen3NextAttention(args)
        self.input_layernorm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(args)
        else:
            self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        residual = x
        if self.is_linear:
            attn_out = self.linear_attn(self.input_layernorm(x))
        else:
            attn_out = self.self_attn(self.input_layernorm(x), mask=mask)
        x = residual + attn_out
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3NextModel(nn.Module):
    def __init__(self, args: Qwen3NextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3NextDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        hidden = self.embed_tokens(inputs)
        if mask is None:
            seq_len = hidden.shape[1]
            mask = build_causal_mask(seq_len, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, mask=mask)
        return self.norm(hidden)


class Qwen3NextForCausalLM(nn.Module):
    def __init__(self, args: Qwen3NextConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = self.model(inputs)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    @property
    def layers(self) -> Sequence[nn.Module]:
        return self.model.layers


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------


def ensure_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    local = snapshot_download(
        path_or_repo,
        allow_patterns=[
            "*.json",
            "model*.safetensors",
            "*.model",
            "*.tiktoken",
            "*.txt",
        ],
    )
    return Path(local)


def load_config(model_path: Path) -> Dict[str, Any]:
    with open(model_path / "config.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_weights(model_path: Path) -> Dict[str, mx.array]:
    weights: Dict[str, mx.array] = {}
    for wf in sorted(model_path.glob("model*.safetensors")):
        weights.update(mx.load(wf))
    if not weights:
        raise FileNotFoundError(f"No weight shards named model*.safetensors in {model_path}")
    return weights


def apply_quantization(model: nn.Module, config: Dict[str, Any], weights: Dict[str, mx.array]) -> None:
    quant_cfg = config.get("quantization")
    if not quant_cfg:
        return

    group_size = quant_cfg.get("group_size", 64)
    bits = quant_cfg.get("bits", 4)
    mode = quant_cfg.get("mode", "affine")

    def predicate(path: str, module: nn.Module):
        specific = quant_cfg.get(path)
        if isinstance(specific, dict):
            return specific
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in weights

    quantize_layers(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        class_predicate=predicate,
    )


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def build_causal_mask(seq_len: int, dtype: mx.Dtype) -> mx.array:
    indices = mx.arange(seq_len)
    mask = indices[:, None] >= indices[None, :]
    mask = mx.astype(mask, mx.bool_)
    return mx.expand_dims(mx.expand_dims(mask, 0), 0)


def select_next_token(logits: mx.array, temperature: float, top_p: float, rng: np.random.Generator) -> int:
    logits = logits.astype(mx.float32)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    probs_np = np.array(probs, dtype=np.float64)
    if top_p < 1.0:
        order = np.argsort(probs_np)[::-1]
        sorted_probs = probs_np[order]
        cumulative = np.cumsum(sorted_probs)
        cutoff = cumulative <= top_p
        if not np.any(cutoff):
            cutoff = cumulative <= min(top_p + 1e-6, 1.0)
        sorted_probs = sorted_probs[cutoff]
        order = order[cutoff]
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = rng.choice(order, p=sorted_probs)
    else:
        probs_np = probs_np / probs_np.sum()
        choice = rng.choice(len(probs_np), p=probs_np)
    return int(choice)


def generate_text(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    seed: Optional[int] = None,
) -> str:
    require_mlx()
    encoded = tokenizer(prompt, return_tensors="np")
    input_ids = encoded["input_ids"][0].tolist()
    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        model_input = mx.array([input_ids], dtype=mx.int32)
        logits = model(model_input)
        logits = logits[:, -1, :].reshape(-1)
        token_id = select_next_token(logits, temperature, top_p, rng)
        input_ids.append(int(token_id))
        if stream:
            piece = tokenizer.decode([token_id], skip_special_tokens=True)
            if piece:
                print(piece, end="", flush=True)
        if token_id == tokenizer.eos_token_id:
            break
    if stream:
        print("\n--- end ---")
    return tokenizer.decode(input_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def describe_parameters(model: nn.Module) -> Iterable[Tuple[str, Tuple[int, ...], float]]:
    require_mlx()
    flat = dict(tree_flatten(model.parameters()))
    for name, tensor in flat.items():
        size_mb = tensor.size * tensor.dtype.itemsize / (1024**2)
        yield name, tensor.shape, size_mb


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF repo or local path")
    parser.add_argument("--prompt", default="Qwen3 Next says:", help="Prompt to feed the model")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Number of new tokens to sample")
    parser.add_argument("--temperature", type=float, default=0.7, help="Softmax temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling cutoff")
    parser.add_argument("--seed", type=int, help="Random seed for sampling")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they arrive")
    parser.add_argument("--show-params", action="store_true", help="Print parameter inventory")
    parser.add_argument("--output-json", type=Path, help="Optional path to dump generation metadata")
    return parser.parse_args(argv)


def load_model_and_tokenizer(model_ref: str) -> Tuple[nn.Module, AutoTokenizer, Dict[str, Any]]:
    require_mlx()
    model_path = ensure_model_path(model_ref)
    config_dict = load_config(model_path)
    config = Qwen3NextConfig.from_dict(config_dict)
    model = Qwen3NextForCausalLM(config)
    weights = load_weights(model_path)
    apply_quantization(model, config_dict, weights)
    model.load_weights(list(weights.items()), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer, config_dict


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    model, tokenizer, config = load_model_and_tokenizer(args.model)

    if args.show_params:
        total = 0.0
        for name, shape, size_mb in describe_parameters(model):
            total += size_mb
            print(f"{name:60s} shape={tuple(shape)} size={size_mb:8.2f} MB")
        print(f"Total parameter footprint (approx.): {total:8.2f} MB")

    if args.stream:
        print("\n--- Generation (stream) ---")
    completion = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=args.stream,
        seed=args.seed,
    )
    if not args.stream:
        print("\n--- Completion ---")
        print(completion)
        print("--- end ---")

    if args.output_json:
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "settings": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            },
            "config": config,
            "completion": completion,
        }
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Metadata saved to {args.output_json}")


if __name__ == "__main__":
    main()
