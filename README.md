# qwen3-next-from-scratch

Hands-on resources for understanding and running Qwen 3 Next:

- `qwen3-next-from-scratch.ipynb` — PyTorch, from-scratch re-implementation with commentary on Gated DeltaNet, partial RoPE, and multi-token prediction.
- `mlx_qwen3_next.py` — Apple MLX helper to load 4-bit (q4) Qwen 3 Next checkpoints for fast inference on M-series chips.

## Quickstart

1. Install Python deps (PyTorch stack, `einops`, `ipywidgets`, `mlx-lm`).
2. Work through the notebook to understand the architecture.
3. Run `python mlx_qwen3_next.py --prompt "Hello"` to sample from a quantized checkpoint (defaults to `Qwen/Qwen3-1.5B-Instruct`).

Feel free to swap in official hyperparameters or larger checkpoints as you experiment.
