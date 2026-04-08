"""
Diagnostic script to determine if the reversible activation mismatch is from:
  Mode A: Non-deterministic recomputation (G(y1) differs between forward and recompute)
  Mode B: Floating-point (a+b)-b != a (addition/subtraction precision loss)
"""
import torch
import torch.nn as nn
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from reversible_model import ReversibleBlock, GPTConfig, LayerNorm, two_sum

def test_recomputation_determinism(device, dtype, use_deterministic=False):
    """Test whether G(y1) and F(x2) produce identical results when called twice."""
    print(f"\n{'='*60}")
    print(f"TEST: Recomputation determinism")
    print(f"  device={device}, dtype={dtype}, deterministic_algos={use_deterministic}")
    print(f"{'='*60}")

    if use_deterministic:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    torch.manual_seed(42)
    config = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=4, n_embd=128, dropout=0.0, bias=False)
    block = ReversibleBlock(config).to(device=device)
    if dtype == torch.float64:
        block = block.double()
    block.eval()

    x = torch.randn(2, 32, 128, device=device, dtype=dtype)

    with torch.no_grad():
        g_out_1 = block.G(x.clone())
        g_out_2 = block.G(x.clone())
        f_out_1 = block.F(x.clone())
        f_out_2 = block.F(x.clone())

    g_bitwise = torch.equal(g_out_1, g_out_2)
    f_bitwise = torch.equal(f_out_1, f_out_2)
    g_maxdiff = (g_out_1 - g_out_2).abs().max().item()
    f_maxdiff = (f_out_1 - f_out_2).abs().max().item()

    print(f"  G(y) bitwise identical across 2 calls: {g_bitwise} (max diff: {g_maxdiff:.2e})")
    print(f"  F(x) bitwise identical across 2 calls: {f_bitwise} (max diff: {f_maxdiff:.2e})")

    if g_bitwise and f_bitwise:
        print("  -> Mode A (non-determinism) is NOT active for this config")
    else:
        print("  -> Mode A (non-determinism) IS ACTIVE")

    if use_deterministic:
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    return g_bitwise and f_bitwise


def test_addition_subtraction_roundtrip(device, dtype):
    """Test whether (x + f(y)) - f(y) == x, with and without compensated summation."""
    print(f"\n{'='*60}")
    print(f"TEST: Addition/subtraction roundtrip precision")
    print(f"  device={device}, dtype={dtype}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    config = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=4, n_embd=128, dropout=0.0, bias=False)
    block = ReversibleBlock(config).to(device=device)
    if dtype == torch.float64:
        block = block.double()
    block.eval()

    x2 = torch.randn(2, 32, 128, device=device, dtype=dtype)
    x1 = torch.randn(2, 32, 128, device=device, dtype=dtype)

    # --- Without compensation (naive subtraction) ---
    print("  [Without compensation]")
    with torch.no_grad():
        f_x2 = block.F(x2)
        y1 = x1 + f_x2
        g_y1 = block.G(y1)
        y2 = x2 + g_y1

        g_y1_recomputed = block.G(y1)
        x2_naive = y2 - g_y1_recomputed
        f_x2_recomp_naive = block.F(x2_naive)
        x1_naive = y1 - f_x2_recomp_naive

    x2_diff_naive = (x2 - x2_naive).abs().max().item()
    x1_diff_naive = (x1 - x1_naive).abs().max().item()
    print(f"    x2 max abs diff: {x2_diff_naive:.2e}, exact={torch.equal(x2, x2_naive)}")
    print(f"    x1 max abs diff: {x1_diff_naive:.2e}, exact={torch.equal(x1, x1_naive)}")

    # --- With TwoSum compensation ---
    print("  [With TwoSum compensated summation]")
    with torch.no_grad():
        f_x2 = block.F(x2)
        y1, y1_err = two_sum(x1, f_x2)

        g_y1 = block.G(y1)
        y2, y2_err = two_sum(x2, g_y1)

        g_y1_recomputed = block.G(y1)
        x2_approx, sub_err = two_sum(y2, -g_y1_recomputed)
        x2_comp = x2_approx + (sub_err + y2_err)

        f_x2_recomp_comp = block.F(x2_comp)
        x1_approx, sub_err = two_sum(y1, -f_x2_recomp_comp)
        x1_comp = x1_approx + (sub_err + y1_err)

    x2_diff_comp = (x2 - x2_comp).abs().max().item()
    x1_diff_comp = (x1 - x1_comp).abs().max().item()
    print(f"    x2 max abs diff: {x2_diff_comp:.2e}, exact={torch.equal(x2, x2_comp)}")
    print(f"    x1 max abs diff: {x1_diff_comp:.2e}, exact={torch.equal(x1, x1_comp)}")

    if torch.equal(x2, x2_comp) and torch.equal(x1, x1_comp):
        print("  -> TwoSum achieves EXACT reversibility!")
    elif x2_diff_comp < x2_diff_naive:
        print(f"  -> TwoSum improved x2 recovery by {x2_diff_naive/max(x2_diff_comp,1e-30):.1f}x")
    else:
        print(f"  -> TwoSum did not improve (ratio: {x2_diff_naive/max(x2_diff_comp,1e-30):.1f}x)")


def test_layernorm_isolation(device, dtype):
    """Test specifically whether LayerNorm is the culprit by comparing with/without it."""
    print(f"\n{'='*60}")
    print(f"TEST: LayerNorm isolation -- compare roundtrip with LN vs without")
    print(f"  device={device}, dtype={dtype}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(2, 32, 128, device=device, dtype=dtype)

    # Test 1: simple linear (should be exact or near-exact)
    linear = nn.Linear(128, 128, bias=False).to(device=device)
    if dtype == torch.float64:
        linear = linear.double()
    with torch.no_grad():
        f_x = linear(x)
        y = x + f_x
        f_x_recomp = linear(x)
        x_recovered = y - f_x_recomp
    print(f"  Linear only:   roundtrip exact={torch.equal(x, x_recovered)}, max_diff={(x - x_recovered).abs().max().item():.2e}")

    # Test 2: LayerNorm + linear
    ln = LayerNorm(128, bias=False).to(device=device)
    if dtype == torch.float64:
        ln = ln.double()
    with torch.no_grad():
        f_x = linear(ln(x))
        y = x + f_x
        f_x_recomp = linear(ln(x))
        x_recovered = y - f_x_recomp
    print(f"  LN + Linear:   roundtrip exact={torch.equal(x, x_recovered)}, max_diff={(x - x_recovered).abs().max().item():.2e}")

    # Test 3: nn.Identity (no normalization at all)
    with torch.no_grad():
        f_x = linear(x)
        y = x + f_x
        f_x_recomp = linear(x)
        x_recovered = y - f_x_recomp
    print(f"  Identity+Linear: roundtrip exact={torch.equal(x, x_recovered)}, max_diff={(x - x_recovered).abs().max().item():.2e}")

    # Test 4: LayerNorm only (no linear after)
    with torch.no_grad():
        f_x = ln(x)
        y = x + f_x
        f_x_recomp = ln(x)
        x_recovered = y - f_x_recomp
    print(f"  LN only:       roundtrip exact={torch.equal(x, x_recovered)}, max_diff={(x - x_recovered).abs().max().item():.2e}")


def test_multi_layer_error_accumulation(device, dtype, n_layers=4):
    """Test how errors accumulate across multiple reversible layers."""
    print(f"\n{'='*60}")
    print(f"TEST: Multi-layer error accumulation ({n_layers} layers)")
    print(f"  device={device}, dtype={dtype}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    config = GPTConfig(block_size=64, vocab_size=256, n_layer=n_layers, n_head=4, n_embd=128, dropout=0.0, bias=False)
    blocks = nn.ModuleList([ReversibleBlock(config) for _ in range(n_layers)]).to(device=device)
    if dtype == torch.float64:
        blocks = blocks.double()
    for b in blocks:
        b.eval()

    x1 = torch.randn(2, 32, 128, device=device, dtype=dtype)
    x2 = torch.randn(2, 32, 128, device=device, dtype=dtype)

    # Forward: save activations and error terms at each layer boundary
    fwd_acts = [(x1.clone(), x2.clone())]
    err_terms = []
    for block in blocks:
        with torch.no_grad():
            x1, x2, y1_err, y2_err = block(x1, x2)
        fwd_acts.append((x1.clone(), x2.clone()))
        err_terms.append((y1_err.clone(), y2_err.clone()))

    # Backward reconstruction WITHOUT compensation
    print("  [Without compensation]")
    y1, y2 = fwd_acts[-1]
    for i, block in enumerate(reversed(blocks)):
        with torch.no_grad():
            g_y1 = block.G(y1)
            x2_rec = y2 - g_y1
            f_x2 = block.F(x2_rec)
            x1_rec = y1 - f_x2
        layer_idx = n_layers - 1 - i
        fwd_x1, fwd_x2 = fwd_acts[layer_idx]
        d1 = (fwd_x1 - x1_rec).abs().max().item()
        d2 = (fwd_x2 - x2_rec).abs().max().item()
        print(f"    Layer {layer_idx}: x1 max_diff={d1:.2e}, x2 max_diff={d2:.2e}")
        y1, y2 = x1_rec, x2_rec

    # Backward reconstruction WITH TwoSum compensation
    print("  [With TwoSum compensated summation]")
    y1, y2 = fwd_acts[-1]
    for i, block in enumerate(reversed(blocks)):
        layer_idx = n_layers - 1 - i
        y1_err, y2_err = err_terms[layer_idx]
        with torch.no_grad():
            g_y1 = block.G(y1)
            x2_approx, sub_err = two_sum(y2, -g_y1)
            x2_rec = x2_approx + (sub_err + y2_err)
            f_x2 = block.F(x2_rec)
            x1_approx, sub_err = two_sum(y1, -f_x2)
            x1_rec = x1_approx + (sub_err + y1_err)
        fwd_x1, fwd_x2 = fwd_acts[layer_idx]
        d1 = (fwd_x1 - x1_rec).abs().max().item()
        d2 = (fwd_x2 - x2_rec).abs().max().item()
        print(f"    Layer {layer_idx}: x1 max_diff={d1:.2e}, x2 max_diff={d2:.2e}")
        y1, y2 = x1_rec, x2_rec


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    for dtype in [torch.float32, torch.float64]:
        dtype_name = "float32" if dtype == torch.float32 else "float64"
        print(f"\n\n{'#'*70}")
        print(f"# DTYPE: {dtype_name}")
        print(f"{'#'*70}")

        tests = [
            ("recomp_determ", lambda d=dtype: test_recomputation_determinism(device, d, use_deterministic=False)),
            ("roundtrip", lambda d=dtype: test_addition_subtraction_roundtrip(device, d)),
            ("ln_isolation", lambda d=dtype: test_layernorm_isolation(device, d)),
            ("multi_layer", lambda d=dtype: test_multi_layer_error_accumulation(device, d, n_layers=4)),
        ]
        if device.type == "cuda":
            tests.insert(1, ("recomp_determ_det", lambda d=dtype: test_recomputation_determinism(device, d, use_deterministic=True)))

        for name, test_fn in tests:
            try:
                test_fn()
            except Exception as e:
                print(f"\n  TEST '{name}' FAILED with error: {e}")
                if "deterministic" in str(e).lower():
                    torch.use_deterministic_algorithms(False)
