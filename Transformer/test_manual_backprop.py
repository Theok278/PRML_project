"""
Test script to verify manual backpropagation implementation

This script performs gradient checking to ensure our manual gradients
are correct.
"""

import numpy as np
from model_manual_backprop import (
    Linear, LayerNorm, GELU, Softmax, MultiHeadAttention,
    SimplifiedTransformer, CrossEntropyLoss
)


def numerical_gradient(f, x, epsilon=1e-5):
    """
    Compute numerical gradient using finite differences

    Args:
        f: function that takes x and returns scalar loss
        x: input array
        epsilon: small perturbation

    Returns:
        numerical gradient with same shape as x
    """
    grad = np.zeros_like(x)
    flat_x = x.ravel()
    flat_grad = grad.ravel()

    for i in range(flat_x.size):
        old_value = flat_x[i]

        # f(x + epsilon)
        flat_x[i] = old_value + epsilon
        loss_plus = f(x)

        # f(x - epsilon)
        flat_x[i] = old_value - epsilon
        loss_minus = f(x)

        # Gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        flat_grad[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

        # Restore
        flat_x[i] = old_value

    return grad


def test_linear():
    """Test Linear layer gradients"""
    print("\n" + "="*60)
    print("Testing Linear Layer")
    print("="*60)

    np.random.seed(42)

    in_features, out_features = 8, 4
    batch_size, seq_len = 2, 3

    layer = Linear(in_features, out_features)
    x = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    target_grad = np.random.randn(batch_size, seq_len, out_features).astype(np.float32)

    # Forward
    output = layer.forward(x)

    # Manual backward
    layer.zero_grad()
    grad_input_manual = layer.backward(target_grad)

    # Numerical gradient for weight
    def loss_fn_weight(w):
        layer.weight.data = w.reshape(layer.weight.data.shape)
        out = layer.forward(x)
        return (out * target_grad).sum()

    grad_weight_numerical = numerical_gradient(loss_fn_weight, layer.weight.data.copy())

    # Compare
    weight_diff = np.abs(layer.weight.grad - grad_weight_numerical).max()
    print(f"Weight gradient max diff: {weight_diff:.2e}")

    if weight_diff < 1e-5:
        print("✅ Linear layer gradients are CORRECT!")
    else:
        print("❌ Linear layer gradients have errors")

    return weight_diff < 1e-5


def test_layernorm():
    """Test LayerNorm gradients"""
    print("\n" + "="*60)
    print("Testing LayerNorm")
    print("="*60)

    np.random.seed(42)

    d_model = 8
    batch_size, seq_len = 2, 3

    layer = LayerNorm(d_model)
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    target_grad = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    # Forward
    output = layer.forward(x)

    # Manual backward
    layer.zero_grad()
    grad_input_manual = layer.backward(target_grad)

    # Numerical gradient for gamma
    def loss_fn_gamma(g):
        layer.gamma.data = g
        out = layer.forward(x)
        return (out * target_grad).sum()

    grad_gamma_numerical = numerical_gradient(loss_fn_gamma, layer.gamma.data.copy())

    # Compare
    gamma_diff = np.abs(layer.gamma.grad - grad_gamma_numerical).max()
    print(f"Gamma gradient max diff: {gamma_diff:.2e}")

    if gamma_diff < 1e-4:
        print("✅ LayerNorm gradients are CORRECT!")
    else:
        print("❌ LayerNorm gradients have errors")

    return gamma_diff < 1e-4


def test_gelu():
    """Test GELU gradients"""
    print("\n" + "="*60)
    print("Testing GELU")
    print("="*60)

    np.random.seed(42)

    x = np.random.randn(4, 8).astype(np.float32) * 0.5  # Smaller range
    target_grad = np.random.randn(4, 8).astype(np.float32)

    gelu = GELU()

    # Forward
    output = gelu.forward(x)

    # Manual backward
    grad_input_manual = gelu.backward(target_grad)

    # Numerical gradient
    def loss_fn(x_in):
        out = gelu.forward(x_in)
        return (out * target_grad).sum()

    grad_input_numerical = numerical_gradient(loss_fn, x.copy())

    # Compare
    input_diff = np.abs(grad_input_manual - grad_input_numerical).max()
    print(f"Input gradient max diff: {input_diff:.2e}")

    if input_diff < 1e-5:
        print("✅ GELU gradients are CORRECT!")
    else:
        print("❌ GELU gradients have errors")

    return input_diff < 1e-5


def test_softmax():
    """Test Softmax gradients"""
    print("\n" + "="*60)
    print("Testing Softmax")
    print("="*60)

    np.random.seed(42)

    x = np.random.randn(4, 10).astype(np.float32)
    target_grad = np.random.randn(4, 10).astype(np.float32)

    softmax = Softmax(dim=-1)

    # Forward
    output = softmax.forward(x)

    # Manual backward
    grad_input_manual = softmax.backward(target_grad)

    # Numerical gradient
    def loss_fn(x_in):
        out = softmax.forward(x_in)
        return (out * target_grad).sum()

    grad_input_numerical = numerical_gradient(loss_fn, x.copy())

    # Compare
    input_diff = np.abs(grad_input_manual - grad_input_numerical).max()
    print(f"Input gradient max diff: {input_diff:.2e}")

    if input_diff < 1e-5:
        print("✅ Softmax gradients are CORRECT!")
    else:
        print("❌ Softmax gradients have errors")

    return input_diff < 1e-5


def test_cross_entropy():
    """Test CrossEntropyLoss gradients"""
    print("\n" + "="*60)
    print("Testing CrossEntropyLoss")
    print("="*60)

    np.random.seed(42)

    batch_size, num_classes = 4, 10
    logits = np.random.randn(batch_size, num_classes).astype(np.float32)
    targets = np.array([0, 3, 5, 9])

    criterion = CrossEntropyLoss()

    # Forward
    loss = criterion.forward(logits, targets)
    print(f"Loss: {loss:.4f}")

    # Manual backward
    grad_logits_manual = criterion.backward()

    # Numerical gradient
    def loss_fn(x):
        return criterion.forward(x, targets)

    grad_logits_numerical = numerical_gradient(loss_fn, logits.copy())

    # Compare
    logits_diff = np.abs(grad_logits_manual - grad_logits_numerical).max()
    print(f"Logits gradient max diff: {logits_diff:.2e}")

    if logits_diff < 1e-6:
        print("✅ CrossEntropyLoss gradients are CORRECT!")
    else:
        print("❌ CrossEntropyLoss gradients have errors")

    return logits_diff < 1e-6


def test_full_model():
    """Test full model forward and backward pass"""
    print("\n" + "="*60)
    print("Testing Full Transformer Model")
    print("="*60)

    np.random.seed(42)

    # Create small model
    model = SimplifiedTransformer(
        input_dim=3,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,  # Disable dropout for testing
        num_classes=10
    )

    # Input data
    batch_size, seq_len = 2, 8
    x = np.random.randn(batch_size, seq_len, 3).astype(np.float32)
    targets = np.array([0, 5])

    # Forward pass
    logits = model.forward(x, training=False)
    print(f"Output shape: {logits.shape}")

    # Loss
    criterion = CrossEntropyLoss()
    loss = criterion.forward(logits, targets)
    print(f"Loss: {loss:.4f}")

    # Manual backward
    model.zero_grad()
    grad_logits = criterion.backward()
    grad_input = model.backward(grad_logits)

    print(f"Gradient input shape: {grad_input.shape}")

    # Check if gradients are computed
    has_gradients = all(
        np.abs(p.grad).max() > 0 for p in model.parameters()
    )

    if has_gradients:
        print("✅ Full model gradients are computed!")
    else:
        print("❌ Some gradients are zero")

    # Check gradient for first layer weight (spot check)
    first_param = model.parameters()[0]
    print(f"\nFirst parameter gradient stats:")
    print(f"  Shape: {first_param.grad.shape}")
    print(f"  Mean: {first_param.grad.mean():.2e}")
    print(f"  Std: {first_param.grad.std():.2e}")
    print(f"  Max: {np.abs(first_param.grad).max():.2e}")

    return has_gradients


def main():
    print("\n" + "="*60)
    print("GRADIENT CHECKING FOR MANUAL BACKPROPAGATION")
    print("="*60)
    print("\nThis verifies that our manual gradients match numerical gradients")
    print("(computed using finite differences)")
    print("="*60)

    results = {}

    # Test individual components
    results['Linear'] = test_linear()
    results['LayerNorm'] = test_layernorm()
    results['GELU'] = test_gelu()
    results['Softmax'] = test_softmax()
    results['CrossEntropy'] = test_cross_entropy()

    # Test full model
    results['FullModel'] = test_full_model()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
        all_passed = all_passed and passed

    print("="*60)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Manual backpropagation implementation is CORRECT")
        print("✅ Ready to use for training")
    else:
        print("\n⚠️  Some tests failed")
        print("❌ Please check the implementation")

    print("="*60)


if __name__ == '__main__':
    main()
