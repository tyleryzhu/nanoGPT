"""
Gradient matching test: compare gradients from vanilla_backward=True (standard autograd)
vs vanilla_backward=False (reversible backprop with TwoSum compensation).

If the TwoSum fix works correctly, the gradients should be identical or near-identical
because the activations are now reconstructed exactly.
"""
import torch
import copy
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from reversible_model import GPTConfig, ReversibleGPT

def test_gradient_match(device, dtype, batch_size=4, seq_len=64):
    print(f"\n{'='*70}")
    print(f"GRADIENT MATCHING TEST: device={device}, dtype={dtype}")
    print(f"{'='*70}")

    torch.manual_seed(1337)
    config = GPTConfig(
        block_size=64, vocab_size=256, n_layer=4, n_head=4,
        n_embd=128, dropout=0.0, bias=False
    )

    model_vanilla = ReversibleGPT(config).to(device)
    if dtype == torch.float64:
        model_vanilla = model_vanilla.double()
    model_vanilla.use_vanilla_backward = True

    model_rev = copy.deepcopy(model_vanilla)
    model_rev.use_vanilla_backward = False

    # Verify weights are identical
    for (n1, p1), (n2, p2) in zip(model_vanilla.named_parameters(), model_rev.named_parameters()):
        assert torch.equal(p1, p2), f"Weight mismatch at {n1}"

    torch.manual_seed(42)
    dummy_in = torch.randint(0, 256, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 256, (batch_size, seq_len), device=device)

    # Forward + backward with vanilla (standard autograd)
    logits_v, loss_v = model_vanilla(dummy_in, targets=labels)
    loss_v.backward()

    # Forward + backward with reversible backprop
    logits_r, loss_r = model_rev(dummy_in, targets=labels)
    loss_r.backward()

    # Compare losses
    loss_diff = abs(loss_v.item() - loss_r.item())
    print(f"\n  Loss (vanilla):    {loss_v.item():.10f}")
    print(f"  Loss (reversible): {loss_r.item():.10f}")
    print(f"  Loss diff:         {loss_diff:.2e}")

    # Compare logits
    logit_maxdiff = (logits_v - logits_r).abs().max().item()
    print(f"  Logit max diff:    {logit_maxdiff:.2e}")

    # Compare gradients parameter by parameter
    print(f"\n  {'Parameter':<45} {'Max Grad Diff':>15} {'Rel Diff':>15} {'Match':>8}")
    print(f"  {'-'*88}")

    all_match = True
    max_diff_overall = 0.0
    max_rel_overall = 0.0

    for (n1, p1), (n2, p2) in zip(model_vanilla.named_parameters(), model_rev.named_parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        if p1.grad is None or p2.grad is None:
            print(f"  {n1:<45} {'MISSING GRAD':>15}")
            all_match = False
            continue

        abs_diff = (p1.grad - p2.grad).abs().max().item()
        grad_norm = p1.grad.abs().max().item()
        rel_diff = abs_diff / (grad_norm + 1e-30)
        match = torch.equal(p1.grad, p2.grad)

        max_diff_overall = max(max_diff_overall, abs_diff)
        max_rel_overall = max(max_rel_overall, rel_diff)

        if not match:
            all_match = False

        status = "EXACT" if match else "CLOSE" if abs_diff < 1e-5 else "DIFF"
        print(f"  {n1:<45} {abs_diff:>15.2e} {rel_diff:>15.2e} {status:>8}")

    print(f"\n  Overall max abs gradient diff: {max_diff_overall:.2e}")
    print(f"  Overall max rel gradient diff: {max_rel_overall:.2e}")
    if all_match:
        print("  => ALL GRADIENTS MATCH EXACTLY!")
    elif max_diff_overall < 1e-5:
        print("  => Gradients are very close (max diff < 1e-5)")
    else:
        print("  => GRADIENT MISMATCH DETECTED")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    for dtype in [torch.float32, torch.float64]:
        test_gradient_match(device, dtype)
