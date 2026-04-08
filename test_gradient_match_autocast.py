"""
Gradient matching under autocast (bfloat16) with varying layer counts.
Tests the actual training regime to measure gradient errors from
the float32/float64 residual + autocast-matching approach.
"""
import torch
import copy
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from reversible_model import GPTConfig, ReversibleGPT


def test_gradient_match_autocast(device, n_layer, residual_dtype_name, batch_size=4, seq_len=64):
    """Test gradient matching under autocast with configurable layer count."""
    print(f"\n{'='*70}")
    print(f"GRADIENT MATCH: {n_layer} layers, autocast=bfloat16, residual={residual_dtype_name}")
    print(f"{'='*70}")

    torch.manual_seed(1337)
    config = GPTConfig(
        block_size=64, vocab_size=256, n_layer=n_layer, n_head=4,
        n_embd=128, dropout=0.0, bias=False
    )

    model_vanilla = ReversibleGPT(config).to(device)
    model_vanilla.use_vanilla_backward = True

    model_rev = copy.deepcopy(model_vanilla)
    model_rev.use_vanilla_backward = False

    torch.manual_seed(42)
    dummy_in = torch.randint(0, 256, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 256, (batch_size, seq_len), device=device)

    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # Forward + backward with vanilla (standard autograd, under autocast)
    with amp_ctx:
        logits_v, loss_v = model_vanilla(dummy_in, targets=labels)
    loss_v.backward()

    # Forward + backward with reversible backprop (under autocast)
    with amp_ctx:
        logits_r, loss_r = model_rev(dummy_in, targets=labels)
    loss_r.backward()

    loss_diff = abs(loss_v.item() - loss_r.item())
    logit_diff = (logits_v.float() - logits_r.float()).abs().max().item()
    print(f"  Loss (vanilla):    {loss_v.item():.10f}")
    print(f"  Loss (reversible): {loss_r.item():.10f}")
    print(f"  Loss diff:         {loss_diff:.2e}")
    print(f"  Logit max diff:    {logit_diff:.2e}")

    max_abs = 0.0
    max_rel = 0.0
    all_exact = True
    worst_param = ""

    for (n1, p1), (n2, p2) in zip(model_vanilla.named_parameters(), model_rev.named_parameters()):
        if p1.grad is None or p2.grad is None:
            continue
        abs_diff = (p1.grad - p2.grad).abs().max().item()
        grad_norm = p1.grad.abs().max().item()
        rel_diff = abs_diff / (grad_norm + 1e-30)

        if abs_diff > max_abs:
            max_abs = abs_diff
            worst_param = n1

        if not torch.equal(p1.grad, p2.grad):
            all_exact = False

        max_rel = max(max_rel, rel_diff)

    print(f"\n  Overall max abs gradient diff: {max_abs:.2e}")
    print(f"  Overall max rel gradient diff: {max_rel:.2e}")
    print(f"  Worst parameter: {worst_param}")
    if all_exact:
        print("  => ALL GRADIENTS MATCH EXACTLY!")
    elif max_abs < 1e-5:
        print("  => Gradients very close (max diff < 1e-5)")
    elif max_abs < 1e-3:
        print("  => Gradients close (max diff < 1e-3)")
    else:
        print("  => GRADIENT MISMATCH DETECTED (max diff >= 1e-3)")

    return max_abs, max_rel


if __name__ == "__main__":
    import reversible_model as rm
    device = torch.device("cuda")
    print(f"Device: {device}, PyTorch: {torch.__version__}")

    print("\n" + "#"*70)
    print("# RESIDUAL_DTYPE = float32")
    print("#"*70)
    rm.RESIDUAL_DTYPE = torch.float32
    for n_layer in [4, 8, 12]:
        test_gradient_match_autocast(device, n_layer, "float32")

    print("\n" + "#"*70)
    print("# RESIDUAL_DTYPE = float64")
    print("#"*70)
    rm.RESIDUAL_DTYPE = torch.float64
    for n_layer in [4, 8, 12]:
        test_gradient_match_autocast(device, n_layer, "float64")
