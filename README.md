# qwen3-next-from-scratch

Hands-on resources for understanding and running Qwen 3 Next:

- `qwen3-next-from-scratch.ipynb` — PyTorch, from-scratch re-implementation with commentary on Gated DeltaNet, zero-centered RMSNorm, gated attention, partial RoPE, and multi-token prediction.
- `mlx_qwen3_next.py` — Pure-MLX loader for the 4-bit Qwen3 Next 80B A3B checkpoint
  published under `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`. The script wires
  up the architecture, applies quantization, and streams text generation directly
  with MLX arrays.

The helper defaults to the 80B A3B instruct checkpoint; expect ~40 GB of q4 weights, so the M3 Ultra with 96 GB unified memory is a good fit.
## Quickstart

1. Install Python deps (`huggingface_hub`, `transformers`, `numpy`) and the MLX
   runtime wheel for Apple Silicon (see the [MLX documentation](https://github.com/apple/mlx) for installation).
2. Work through the notebook to understand the architecture.
3. Run `python mlx_qwen3_next.py --prompt "Hello"` to sample from the quantized
   checkpoint (defaults to `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`).

Feel free to swap in official hyperparameters or larger checkpoints as you experiment.
