#!/usr/bin/env python3
# CAR Mechanism Discrimination Test - Distinguish Noise Recognition vs Noise Averaging

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CARConfig
from src.car_model import CompleteCARModel


def generate_structured_noise(X, noise_level=1e12):
    """Generate structured noise"""
    noise = np.zeros_like(X)
    noise[:, 0] = np.random.randn(len(X)) * noise_level
    noise[:, 1] = np.random.randn(len(X)) * noise_level * 0.5
    return noise


def test_structured_vs_random_noise():
    print("\nTest 1: Structured Noise vs Random Noise")
    print("-" * 60)
    
    X_train = np.random.randn(500, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1)
    
    config = CARConfig(
        KB_CAPACITY=100,
        KB_MERGE_THRESHOLD=0.15,
        DIVERSITY_BONUS=0.3
    )
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(200, 20)
    
    # 结构化噪声：某些维度有噪声，其他维度干净
    X_structured = X_test.copy()
    X_structured[:, 5:] += np.random.randn(200, 15) * 1e12
    
    # 随机噪声：所有维度都有噪声
    X_random = X_test + np.random.randn(200, 20) * 1e12
    
    pred_structured = np.array([model.predict(x) for x in X_structured])
    pred_random = np.array([model.predict(x) for x in X_random])
    
    std_structured = np.std(pred_structured)
    std_random = np.std(pred_random)
    
    print(f"Structured noise prediction std: {std_structured:.4f}")
    print(f"Random noise prediction std: {std_random:.4f}")
    print(f"Difference: {abs(std_structured - std_random):.4f}")
    
    return std_structured, std_random


def test_dimensional_sensitivity():
    print("\nTest 2: Dimensional Sensitivity (Sublinear Scaling Test)")
    print("-" * 60)
    
    X_train = np.random.randn(500, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1)
    
    config = CARConfig(
        KB_CAPACITY=100,
        KB_MERGE_THRESHOLD=0.15,
        DIVERSITY_BONUS=0.3
    )
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(200, 20)
    
    results = []
    for dim in [1, 5, 10, 15, 20]:
        X_noisy = X_test.copy()
        X_noisy[:, :dim] += np.random.randn(200, dim) * 1e12
        
        predictions = np.array([model.predict(x) for x in X_noisy])
        std = np.std(predictions)
        results.append((dim, std))
        
        print(f"Noise dimension {dim:2d}: std = {std:.4f}")
    
    # Calculate scaling factor
    std_change = results[-1][1] - results[0][1]
    dim_change = results[-1][0] - results[0][0]
    scaling_factor = std_change / dim_change if dim_change > 0 else 0
    
    print(f"\nScaling analysis:")
    print(f"  Standard deviation change: {std_change:.4f}")
    print(f"  Dimension change: {dim_change}")
    print(f"  Scaling factor: {scaling_factor:.4f}")
    
    if scaling_factor < 0.02:
        print("  ✓ Sublinear scaling (supports noise recognition)")
    else:
        print("  ⚠ Linear or superlinear scaling")
    
    return results, scaling_factor


def test_adversarial_noise_with_hidden_signal():
    print("\nTest 3: Adversarial Noise + Hidden Signal")
    print("-" * 60)
    
    X_train = np.random.randn(500, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1)
    
    config = CARConfig(
        KB_CAPACITY=100,
        KB_MERGE_THRESHOLD=0.15,
        DIVERSITY_BONUS=0.3
    )
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(200, 20)
    y_test = np.sum(np.sin(X_test[:, :3]), axis=1)
    
    # 对抗性噪声：前3个特征有隐藏信号，其他维度有极端噪声
    X_noisy = X_test.copy()
    X_noisy[:, 3:] += np.random.randn(200, 17) * 1e12
    
    predictions = np.array([model.predict(x) for x in X_noisy])
    mse = np.mean((predictions - y_test) ** 2)
    
    print(f"MSE under adversarial noise: {mse:.4f}")
    
    if mse < 1.0:
        print("✓ CAR recognized hidden signal")
    elif mse < 5.0:
        print("⚠ CAR partially recognized hidden signal")
    else:
        print("✗ CAR was interfered by noise")
    
    return mse


def test_noise_pattern_differentiation():
    print("\nTest 4: Noise Pattern Differentiation")
    print("-" * 60)
    
    X_train = np.random.randn(500, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1)
    
    config = CARConfig(
        KB_CAPACITY=100,
        KB_MERGE_THRESHOLD=0.15,
        DIVERSITY_BONUS=0.3
    )
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    X_test = np.random.randn(200, 20)
    
    noise_patterns = [
        ("Single dimension noise", lambda X: X + np.random.randn(200, 20) * np.array([1e12] + [0] * 19)),
        ("Two dimension noise", lambda X: X + np.random.randn(200, 20) * np.array([1e12, 1e12] + [0] * 18)),
        ("Three dimension noise", lambda X: X + np.random.randn(200, 20) * np.array([1e12, 1e12, 1e12] + [0] * 17)),
    ]
    
    results = []
    for name, noise_fn in noise_patterns:
        X_noisy = noise_fn(X_test)
        predictions = np.array([model.predict(x) for x in X_noisy])
        std = np.std(predictions)
        results.append((name, std))
        print(f"{name}: std = {std:.4f}")
    
    return results


def run_mechanism_discrimination_test():
    print("=" * 60)
    print("  CAR Mechanism Discrimination Test")
    print("=" * 60)
    print("\nGoal: Verify CAR uses noise recognition instead of noise averaging")
    
    std_structured, std_random = test_structured_vs_random_noise()
    dim_results, scaling_factor = test_dimensional_sensitivity()
    adversarial_mse = test_adversarial_noise_with_hidden_signal()
    pattern_results = test_noise_pattern_differentiation()
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    print(f"\nStructured vs Random Noise:")
    print(f"  Structured: {std_structured:.4f}, Random: {std_random:.4f}")
    diff = abs(std_structured - std_random)
    
    if diff > 0.05:
        print(f"  ✓ CAR is sensitive to different noise patterns (difference: {diff:.4f})")
    elif diff > 0.02:
        print(f"  ⚠ CAR has some noise pattern differentiation (difference: {diff:.4f})")
    else:
        print(f"  ✗ CAR has insufficient noise pattern differentiation (difference: {diff:.4f})")
    
    print(f"\nDimensional Sensitivity:")
    print(f"  Standard deviation change from 1 to 20 dimensions: {dim_results[-1][1] - dim_results[0][1]:.4f}")
    print(f"  Scaling factor: {scaling_factor:.4f}")
    
    print(f"\nAdversarial Noise:")
    print(f"  MSE: {adversarial_mse:.4f}")
    
    return {
        'structured_std': std_structured,
        'random_std': std_random,
        'dimensional_results': dim_results,
        'scaling_factor': scaling_factor,
        'adversarial_mse': adversarial_mse,
        'pattern_results': pattern_results
    }


if __name__ == "__main__":
    results = run_mechanism_discrimination_test()
    sys.exit(0)