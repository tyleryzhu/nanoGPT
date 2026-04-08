"""
Diagnose reversibility errors specifically under autocast (mixed precision),
which is the actual training regime. Tests three things:
1. Does autocast cause Mode A (non-deterministic recomputation)?
2. What are the error magnitudes under bfloat16 vs float32 residuals?
3. Does keeping residuals in float32 while functions run in bfloat16 fix it?
"""
import torch
import torch.nn as nn
from contextlib import nullcontext
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from reversible_model import ReversibleBlock, GPTConfig, LayerNorm, _to_residual_dtype


def test_autocast_recomputation(device):
    """Test if G(y1) gives different results when called under autocast vs not."""
    print(f"\n{'='*70}")
    print("TEST: Autocast forward vs no-autocast backward recomputation")
    print(f"{'='*70}")

    torch.manual_seed(42)
    config = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=4,
                       n_embd=128, dropout=0.0, bias=False)
    block = ReversibleBlock(config).to(device)
    block.eval()

    x = torch.randn(2, 32, 128, device=device, dtype=torch.float32)

    # Simulate forward: G(y1) under autocast
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        g_autocast = block.G(x).clone()

    # Simulate backward: G(y1) without autocast
    g_no_autocast = block.G(x).clone()

    match = torch.equal(g_autocast, g_no_autocast)
    diff = (g_autocast.float() - g_no_autocast.float()).abs().max().item()
    print(f"  G(y1) autocast vs no-autocast match: {match}")
    print(f"  Max diff: {diff:.2e}")
    print(f"  G autocast dtype: {g_autocast.dtype}, G no-autocast dtype: {g_no_autocast.dtype}")
    if not match:
        print("  -> MODE A IS ACTIVE under autocast! Forward/backward precision mismatch!")


def test_bfloat16_roundtrip(device):
    """Test (a+b)-b error when everything is bfloat16."""
    print(f"\n{'='*70}")
    print("TEST: Roundtrip errors under bfloat16 (simulating autocast training)")
    print(f"{'='*70}")

    torch.manual_seed(42)
    config = GPTConfig(block_size=64, vocab_size=256, n_layer=1, n_head=4,
                       n_embd=128, dropout=0.0, bias=False)
    block = ReversibleBlock(config).to(device)
    block.eval()

    x2_f32 = torch.randn(2, 32, 128, device=device, dtype=torch.float32)
    x1_f32 = torch.randn(2, 32, 128, device=device, dtype=torch.float32)

    # Case 1: Everything in bfloat16 (current behavior under autocast)
    print("\n  [Case 1: All bfloat16 residuals]")
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        x1_bf = x1_f32.bfloat16()
        x2_bf = x2_f32.bfloat16()
        f_x2 = block.F(x2_bf)
        y1 = x1_bf + f_x2
        g_y1 = block.G(y1)
        y2 = x2_bf + g_y1

    # Reconstruct under autocast (same context)
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        g_y1_re = block.G(y1)
        x2_rec = y2 - g_y1_re
        f_x2_re = block.F(x2_rec)
        x1_rec = y1 - f_x2_re

    g_match = torch.equal(g_y1, g_y1_re)
    x2_diff = (x2_bf.float() - x2_rec.float()).abs().max().item()
    x1_diff = (x1_bf.float() - x1_rec.float()).abs().max().item()
    print(f"    G recompute match: {g_match}")
    print(f"    x2 max abs diff: {x2_diff:.2e}")
    print(f"    x1 max abs diff: {x1_diff:.2e}")
    print(f"    dtypes: y1={y1.dtype}, y2={y2.dtype}")

    # Case 2: Residual stream in float32, functions under autocast
    print("\n  [Case 2: Float32 residuals, bfloat16 functions]")
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        f_x2 = block.F(x1_f32)  # autocast internally, output might be bf16
    y1 = x1_f32 + f_x2.float()  # addition in float32

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        g_y1 = block.G(y1)  # autocast internally
    y2 = x2_f32 + g_y1.float()  # addition in float32

    # Reconstruct with same autocast for recomputation
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        g_y1_re = block.G(y1)
    x2_rec = y2 - g_y1_re.float()  # subtraction in float32

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        f_x2_re = block.F(x2_rec)
    x1_rec = y1 - f_x2_re.float()

    g_match = torch.equal(g_y1, g_y1_re)
    x2_diff = (x2_f32 - x2_rec).abs().max().item()
    x1_diff = (x1_f32 - x1_rec).abs().max().item()
    print(f"    G recompute match: {g_match}")
    print(f"    x2 max abs diff: {x2_diff:.2e}")
    print(f"    x1 max abs diff: {x1_diff:.2e}")
    print(f"    dtypes: y1={y1.dtype}, y2={y2.dtype}")


def test_multilayer_autocast(device, n_layers=4):
    """Test multi-layer error accumulation under autocast."""
    print(f"\n{'='*70}")
    print(f"TEST: Multi-layer ({n_layers}) under autocast")
    print(f"{'='*70}")

    torch.manual_seed(42)
    config = GPTConfig(block_size=64, vocab_size=256, n_layer=n_layers, n_head=4,
                       n_embd=128, dropout=0.0, bias=False)
    blocks = nn.ModuleList([ReversibleBlock(config) for _ in range(n_layers)]).to(device)
    for b in blocks:
        b.eval()

    x1 = torch.randn(2, 32, 128, device=device, dtype=torch.float32)
    x2 = torch.randn(2, 32, 128, device=device, dtype=torch.float32)

    # Case 1: bfloat16 residuals
    print("\n  [bfloat16 residuals]")
    x1_bf, x2_bf = x1.bfloat16(), x2.bfloat16()
    fwd = [(x1_bf.clone(), x2_bf.clone())]
    a, b = x1_bf, x2_bf
    for block in blocks:
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            f_b = block.F(b)
            a_new = a + f_b
            g_a = block.G(a_new)
            b_new = b + g_a
        a, b = a_new, b_new
        fwd.append((a.clone(), b.clone()))

    y1, y2 = fwd[-1]
    for i, block in enumerate(reversed(blocks)):
        idx = n_layers - 1 - i
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            g_y1 = block.G(y1)
        x2_rec = y2 - g_y1
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            f_x2 = block.F(x2_rec)
        x1_rec = y1 - f_x2
        fa, fb = fwd[idx]
        d1 = (fa.float() - x1_rec.float()).abs().max().item()
        d2 = (fb.float() - x2_rec.float()).abs().max().item()
        print(f"    Layer {idx}: x1 diff={d1:.2e}, x2 diff={d2:.2e}")
        y1, y2 = x1_rec, x2_rec

    # Case 2: float32 residuals
    print("\n  [float32 residuals]")
    fwd = [(x1.clone(), x2.clone())]
    a, b = x1.clone(), x2.clone()
    for block in blocks:
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            f_b = block.F(b)
        a_new = a + f_b.float()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            g_a = block.G(a_new)
        b_new = b + g_a.float()
        a, b = a_new, b_new
        fwd.append((a.clone(), b.clone()))

    y1, y2 = fwd[-1]
    for i, block in enumerate(reversed(blocks)):
        idx = n_layers - 1 - i
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            g_y1 = block.G(y1)
        x2_rec = y2 - g_y1.float()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            f_x2 = block.F(x2_rec)
        x1_rec = y1 - f_x2.float()
        fa, fb = fwd[idx]
        d1 = (fa - x1_rec).abs().max().item()
        d2 = (fb - x2_rec).abs().max().item()
        print(f"    Layer {idx}: x1 diff={d1:.2e}, x2 diff={d2:.2e}")
        y1, y2 = x1_rec, x2_rec


def test_multilayer_fixed(device, n_layers=12):
    """Test the actual fix: float32 residuals + same autocast for recomputation."""
    print(f"\n{'='*70}")
    print(f"TEST: Multi-layer ({n_layers}) with FIXED approach")
    print(f"  (float32 residuals + autocast-matched recomputation)")
    print(f"{'='*70}")

    torch.manual_seed(42)
    config = GPTConfig(block_size=64, vocab_size=256, n_layer=n_layers, n_head=4,
                       n_embd=128, dropout=0.0, bias=False)
    blocks = nn.ModuleList([ReversibleBlock(config) for _ in range(n_layers)]).to(device)
    for b in blocks:
        b.eval()

    x1 = torch.randn(2, 32, 128, device=device, dtype=torch.float32)
    x2 = torch.randn(2, 32, 128, device=device, dtype=torch.float32)

    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # Forward with float32 residuals, functions under autocast
    fwd = [(x1.clone(), x2.clone())]
    a, b = x1.clone(), x2.clone()
    for block in blocks:
        with amp_ctx:
            f_b = block.F(b)
        a_new = _to_residual_dtype(a) + _to_residual_dtype(f_b)
        with amp_ctx:
            g_a = block.G(a_new)
        b_new = _to_residual_dtype(b) + _to_residual_dtype(g_a)
        a, b = a_new, b_new
        fwd.append((a.clone(), b.clone()))

    # Backward reconstruction with SAME autocast for recomputation
    y1, y2 = fwd[-1]
    for i, block in enumerate(reversed(blocks)):
        idx = n_layers - 1 - i
        with amp_ctx:
            g_y1 = block.G(y1)
        x2_rec = y2 - _to_residual_dtype(g_y1)
        with amp_ctx:
            f_x2 = block.F(x2_rec)
        x1_rec = y1 - _to_residual_dtype(f_x2)
        fa, fb = fwd[idx]
        d1 = (fa - x1_rec).abs().max().item()
        d2 = (fb - x2_rec).abs().max().item()
        print(f"  Layer {idx}: x1 diff={d1:.2e}, x2 diff={d2:.2e}")
        y1, y2 = x1_rec, x2_rec


def test_multilayer_fp64_residuals(device, n_layers=12):
    """Test float64 residuals: should eliminate amplification errors entirely."""
    print(f"\n{'='*70}")
    print(f"TEST: Multi-layer ({n_layers}) with FLOAT64 residuals")
    print(f"  (double precision residuals + autocast-matched recomputation)")
    print(f"{'='*70}")

    torch.manual_seed(42)
    config = GPTConfig(block_size=64, vocab_size=256, n_layer=n_layers, n_head=4,
                       n_embd=128, dropout=0.0, bias=False)
    blocks = nn.ModuleList([ReversibleBlock(config) for _ in range(n_layers)]).to(device)
    for b in blocks:
        b.eval()

    x1 = torch.randn(2, 32, 128, device=device, dtype=torch.float64)
    x2 = torch.randn(2, 32, 128, device=device, dtype=torch.float64)

    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # Forward with float64 residuals, float32 function inputs (for autocast)
    fwd = [(x1.clone(), x2.clone())]
    a, b = x1.clone(), x2.clone()
    for block in blocks:
        with amp_ctx:
            f_b = block.F(b.float())
        a_new = a + f_b.double()
        with amp_ctx:
            g_a = block.G(a_new.float())
        b_new = b + g_a.double()
        a, b = a_new, b_new
        fwd.append((a.clone(), b.clone()))

    # Backward reconstruction with same autocast + float32 function inputs
    y1, y2 = fwd[-1]
    for i, block in enumerate(reversed(blocks)):
        idx = n_layers - 1 - i
        with amp_ctx:
            g_y1 = block.G(y1.float())
        x2_rec = y2 - g_y1.double()
        with amp_ctx:
            f_x2 = block.F(x2_rec.float())
        x1_rec = y1 - f_x2.double()
        fa, fb = fwd[idx]
        d1 = (fa - x1_rec).abs().max().item()
        d2 = (fb - x2_rec).abs().max().item()
        print(f"  Layer {idx}: x1 diff={d1:.2e}, x2 diff={d2:.2e}")
        y1, y2 = x1_rec, x2_rec


if __name__ == "__main__":
    device = torch.device("cuda")
    print(f"Device: {device}, PyTorch: {torch.__version__}")

    test_autocast_recomputation(device)
    test_bfloat16_roundtrip(device)
    test_multilayer_autocast(device, n_layers=12)
    test_multilayer_fixed(device, n_layers=12)
    test_multilayer_fp64_residuals(device, n_layers=12)
