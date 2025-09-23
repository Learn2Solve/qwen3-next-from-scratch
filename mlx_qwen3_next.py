#!/usr/bin/env python3
"""Minimal MLX runner for 4-bit Qwen 3 Next models.

This helper shows how to load Qwen 3 Next checkpoints with Apple MLX in
`q4` (4-bit) quantization and run quick generations on Apple Silicon.
It complements the PyTorch-from-scratch notebook by offering an optimized
end-to-end path for M-series chips, including the M3 Ultra with unified
memory.

Typical usage (requires `mlx-lm` 0.11+):

    python mlx_qwen3_next.py --model Qwen/Qwen3-1.5B-Instruct --prompt "Hello"

Pass `--list` to inspect available quantized parameter groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

from mlx_lm import generate, load
from mlx_lm.utils import tree_flatten


DEFAULT_MODEL_ID = "Qwen/Qwen3-1.5B-Instruct"


def describe_parameters(model) -> Iterable[Tuple[str, Tuple[int, ...], float]]:
    """Yield (name, shape, size_mb) for each parameter in the MLX tree."""
    flat = dict(tree_flatten(model.parameters()))
    for name, tensor in flat.items():
        size_mb = tensor.size * tensor.itemsize / (1024**2)
        yield name, tensor.shape, size_mb


def load_qwen(model_id: str, quantization: str, device: str):
    """Load a quantized Qwen 3 Next checkpoint with MLX."""
    model, tokenizer = load(model_id, quantization=quantization, device=device)
    return model, tokenizer


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF repo or local path")
    parser.add_argument("--prompt", default="Qwen 3 Next says:", help="Prompt to feed the model")
    parser.add_argument("--quantization", default="q4", help="MLX quantization preset (e.g. q4, q4f16)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Number of new tokens to sample")
    parser.add_argument("--temperature", type=float, default=0.7, help="Softmax temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling cutoff")
    parser.add_argument("--device", default="mps", help="MLX device to target")
    parser.add_argument("--show-params", action="store_true", help="Print parameter inventory")
    parser.add_argument("--list", action="store_true", help="List quantized parameter stats and exit")
    parser.add_argument("--output-json", type=Path, help="Optional path to dump generation metadata")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they are produced")
    return parser.parse_args()


def main() -> None:
    args = cli()
    model, tokenizer = load_qwen(args.model, args.quantization, args.device)

    if args.show_params or args.list:
        total_mb = 0.0
        for name, shape, size_mb in describe_parameters(model):
            total_mb += size_mb
            print(f"{name:60s} shape={tuple(shape)} size={size_mb:7.2f} MB")
        print(f"Total parameter footprint (approx.): {total_mb:7.2f} MB")
        if args.list:
            return

    settings = dict(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=args.stream,
    )
    if args.stream:
        print("\n--- Generation (stream) ---")
        for token in generate(model, tokenizer, args.prompt, **settings):
            print(token, end="", flush=True)
        print("\n--- end ---")
        completion = None
    else:
        completion = generate(model, tokenizer, args.prompt, **settings)
        print("\n--- Completion ---")
        print(completion)
        print("--- end ---")

    if args.output_json:
        payload = {
            "model": args.model,
            "quantization": args.quantization,
            "prompt": args.prompt,
            "settings": settings,
            "completion": completion,
        }
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Metadata saved to {args.output_json}")


if __name__ == "__main__":
    main()
