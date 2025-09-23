# qwen3-next-from-scratch

Hands-on resources for understanding and running Qwen 3 Next:

- `qwen3-next-from-scratch.ipynb` — PyTorch, from-scratch re-implementation with commentary on Gated DeltaNet, partial RoPE, and multi-token prediction.
- `mlx_qwen3_next.py` — Apple MLX helper to load 4-bit (q4) Qwen 3 Next checkpoints for fast inference on M-series chips.

The helper defaults to the 80B A3B instruct checkpoint; expect ~40 GB of q4 weights, so the M3 Ultra with 96 GB unified memory is a good fit.
## Quickstart

1. Install Python deps (PyTorch stack, `einops`, `ipywidgets`, `mlx-lm`).
2. Work through the notebook to understand the architecture.
3. Run `python mlx_qwen3_next.py --prompt "Hello"` to sample from a quantized checkpoint (defaults to `Qwen/Qwen3-Next-80B-A3B-Instruct`).

Feel free to swap in official hyperparameters or larger checkpoints as you experiment.
