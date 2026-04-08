# Changelog

## 2025-04-08 — Fix reversible backprop under mixed-precision training

### Problem

The reversible transformer's activation reconstruction produced large errors
(1e-2 to 1e-3) during mixed-precision (bfloat16 autocast) training, causing
gradient mismatches between the custom reversible backward pass and standard
autograd.

Two root causes were identified:

1. **Autocast context mismatch (Mode A)**: The forward pass ran F/G under
   `torch.amp.autocast(bfloat16)`, but `backward_pass` recomputed them
   *without* autocast. Linear layers therefore used different precisions
   (bfloat16 vs float32), producing ~3.5e-3 differences per recomputation.

2. **Residual precision loss (Mode B)**: Under bfloat16, the residual
   addition/subtraction `(x + f) - f ≠ x` had ~1.6e-2 error per layer.
   Even under float32, the ~1e-7 per-layer error was amplified by the
   attention/MLP Lipschitz constants to ~1e-3 over 8+ layers.

### Solution

Three changes to `reversible_model.py`, requiring **zero per-layer storage**:

- **Save & replay autocast context** — `RevBackProp.forward` captures the
  active autocast state; `backward` replays it via a stored context manager
  so that G/F recomputation is bitwise-identical to the forward.

- **Float64 residual stream** (`RESIDUAL_DTYPE = torch.float64`) — residual
  additions and subtractions are performed in float64 (~1e-16 epsilon).
  F/G still execute in bfloat16 under autocast; only the add/sub uses
  double precision. The stored activation pair (x1, x2) is float64,
  adding ~100 MB for GPT-2 124M — not proportional to `n_layer`.

- **`_fn_input()` helper** — casts float64 residuals to float32 before
  passing to F/G, since float64 tensors bypass CUDA autocast and would
  cause dtype mismatches with bfloat16 weights.

### Removed

- **Knuth TwoSum compensation** — the previous fix stored 2×n_layer error
  tensors (~2.3 GB for GPT-2 124M), negating the memory savings of the
  reversible architecture. Replaced by the float64 residual approach which
  achieves the same (or better) precision with a fixed ~100 MB overhead.

### Validation

| Metric                          | Result                             |
|---------------------------------|------------------------------------|
| Gradient match (4/8/12 layers)  | **Exact** (0.00e+00) under bfloat16 autocast |
| Activation reconstruction       | **Exact** at float64 precision     |
| Training convergence (500 iter) | Rev-backprop = vanilla autograd    |
| `__main__` debug test (float64) | All activations match exactly      |
