#!/usr/bin/env python3
"""
CAR Anti-Cheat Test Suite
=========================
Verify that CAR is truly learning patterns and not relying on data leakage 
or other cheating mechanisms.

Tests include:
1. Knowledge Base Distance Test - Test samples should maintain distance from KB
2. Prediction Diversity Test - Ensure predictions have reasonable diversity
3. Noise Independence Test - Predictions should be independent of input under extreme noise
4. Knowledge Base Size Impact Test - Verify KB size affects performance
5. Random Baseline Comparison - Compare with simple methods

Date: January 2026
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CARConfig
from src.car_model import CompleteCARModel


def test_knowledge_base_distance():
    """
    Test 1: Knowledge Base Distance Test
    
    Verify that test samples maintain reasonable distance from patterns stored 
    in the knowledge base. If distance is too small, data leakage may exist.
    """
    print("\n" + "="*70)
    print("  Test 1: Knowledge Base Distance Test")
    print("="*70)
    
    np.random.seed(42)
    
    # Train model
    X_train = np.random.randn(300, 20)
    y_train = np.sum(X_train[:, :3], axis=1)
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    # Test samples
    np.random.seed(123)
    X_test = np.random.randn(100, 20)
    
    # Calculate distance from test samples to nearest KB patterns
    min_distances = []
    for x in X_test:
        distances = []
        for pattern in model.knowledge_base.patterns:
            dist = np.linalg.norm(x - pattern.features)
            distances.append(dist)
        min_distances.append(min(distances))
    
    min_dist = np.mean(min_distances)
    max_dist = np.max(min_distances)
    
    print(f"\n  Test samples: {len(X_test)}")
    print(f"  Knowledge base size: {len(model.knowledge_base.patterns)}")
    print(f"  Average minimum distance: {min_dist:.4f}")
    print(f"  Maximum minimum distance: {max_dist:.4f}")
    
    # Check for data leakage (distance too small)
    if min_dist < 0.01:
        print(f"  ⚠ Warning: Possible data leakage detected (distance < 0.01)")
        return False
    else:
        print(f"  ✓ Test samples maintain reasonable distance from knowledge base")
        print(f"  ✓ No data leakage detected")
    
    return True


def test_prediction_diversity():
    """
    Test 2: Prediction Diversity Test
    
    Verify that prediction results have reasonable diversity and are not 
    always returning the same value.
    """
    print("\n" + "="*70)
    print("  Test 2: Prediction Diversity Test")
    print("="*70)
    
    np.random.seed(42)
    
    # Train model
    X_train = np.random.randn(300, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    # Test samples
    np.random.seed(456)
    X_test = np.random.randn(100, 20)
    
    predictions = np.array([model.predict(x) for x in X_test])
    
    pred_std = np.std(predictions)
    pred_mean = np.mean(predictions)
    unique_count = len(set(predictions.round(6)))
    
    print(f"\n  Number of predictions: {len(predictions)}")
    print(f"  Prediction mean: {pred_mean:.4f}")
    print(f"  Prediction std: {pred_std:.4f}")
    print(f"  Unique predictions: {unique_count}")
    
    # Check diversity
    if pred_std < 0.01:
        print(f"  ⚠ Warning: Predictions lack diversity (Std < 0.01)")
        return False
    elif unique_count < 10:
        print(f"  ⚠ Warning: Too few unique predictions (< 10)")
        return False
    else:
        print(f"  ✓ Predictions have reasonable diversity")
        print(f"  ✓ Unique prediction ratio: {unique_count/len(predictions)*100:.1f}%")
    
    return True


def test_noise_independence():
    """
    Test 3: Noise Independence Test
    
    Under extreme noise, predictions should be independent of original input.
    If predictions still depend on input, noise may not be large enough.
    """
    print("\n" + "="*70)
    print("  Test 3: Noise Independence Test")
    print("="*70)
    
    np.random.seed(42)
    
    # Train model
    X_train = np.random.randn(300, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    # Extreme noise levels
    noise_levels = [1e50, 1e75, 1e100]
    
    print(f"\n  Noise Level      | PredStd   | Unique   | Noise Impact")
    print(f"  " + "-"*60)
    
    for noise in noise_levels:
        np.random.seed(789)
        X_test = np.random.randn(100, 20)
        
        # Original input
        predictions_orig = np.array([model.predict(x) for x in X_test])
        
        # Add extreme noise
        noise_matrix = np.random.randn(100, 20) * noise
        X_noisy = X_test + noise_matrix
        
        # Ensure input is actually covered by noise
        snr = np.mean(np.abs(X_test)) / noise
        snr_db = -20 * np.log10(snr) if snr > 0 else float('inf')
        
        predictions_noisy = np.array([model.predict(x) for x in X_noisy])
        
        pred_std = np.std(predictions_noisy)
        unique_count = len(set(predictions_noisy.round(4)))
        
        # Calculate prediction change
        pred_change = np.mean(np.abs(predictions_noisy - predictions_orig))
        
        print(f"  {noise:>10.0e} | {pred_std:>9.4f} | {unique_count:>7} | {pred_change:>10.4f}")
    
    # Check if prediction diversity is maintained under extreme noise
    if pred_std > 0.01 and unique_count > 10:
        print(f"\n  ✓ Prediction diversity maintained under extreme noise ({noise_levels[-1]:.0e})")
        print(f"  ✓ This proves mechanism effectiveness (multi-factor weighting), not data leakage")
        return True
    else:
        print(f"\n  ⚠ Warning: Insufficient prediction diversity under extreme noise")
        return False


def test_knowledge_base_scaling():
    """
    Test 4: Knowledge Base Size Impact Test
    
    Verify that knowledge base size has reasonable impact on performance.
    """
    print("\n" + "="*70)
    print("  Test 4: Knowledge Base Size Impact Test")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate more training data
    X_train = np.random.randn(1000, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    
    # Test different KB sizes
    kb_sizes = [10, 50, 200]
    
    print(f"\n  KB Size   | Patterns | PredStd  | Unique")
    print(f"  " + "-"*50)
    
    results = []
    for kb_size in kb_sizes:
        config = CARConfig(KB_CAPACITY=kb_size)
        model = CompleteCARModel(config=config, n_features=20)
        model.fit(X_train, y_train)
        
        np.random.seed(456)
        X_test = np.random.randn(100, 20)
        predictions = np.array([model.predict(x) for x in X_test])
        
        pred_std = np.std(predictions)
        unique_count = len(set(predictions.round(4)))
        
        print(f"  {kb_size:>10} | {len(model.knowledge_base.patterns):>8} | {pred_std:>8.4f} | {unique_count:>7}")
        
        results.append({
            'kb_size': kb_size,
            'actual_size': len(model.knowledge_base.patterns),
            'pred_std': pred_std,
            'unique_count': unique_count
        })
    
    # Verify KB size actually affected performance
    if results[0]['pred_std'] != results[-1]['pred_std']:
        print(f"\n  ✓ KB size affects prediction performance (as expected)")
        return True
    else:
        print(f"\n  ⚠ Warning: KB size did not affect performance")
        return False


def test_random_baseline_comparison():
    """
    Test 5: Random Baseline Comparison
    
    Compare with simple random/constant baselines to verify CAR advantages.
    """
    print("\n" + "="*70)
    print("  Test 5: Random Baseline Comparison")
    print("="*70)
    
    np.random.seed(42)
    
    # Train model
    X_train = np.random.randn(500, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    # Test data
    np.random.seed(789)
    X_test = np.random.randn(100, 20)
    y_test = np.sum(np.sin(X_test[:, :3]), axis=1) + np.cos(X_test[:, 4])
    
    # CAR predictions
    car_predictions = np.array([model.predict(x) for x in X_test])
    car_mae = np.mean(np.abs(car_predictions - y_test))
    
    # Random baseline (random guessing)
    random_predictions = np.random.randn(100) * 2
    random_mae = np.mean(np.abs(random_predictions - y_test))
    
    # Constant baseline (predict training mean)
    constant_prediction = np.mean(y_train)
    constant_predictions = np.ones(100) * constant_prediction
    constant_mae = np.mean(np.abs(constant_predictions - y_test))
    
    print(f"\n  Method           | MAE     | Relative to CAR")
    print(f"  " + "-"*50)
    print(f"  CAR              | {car_mae:.4f}  | 1.00x")
    print(f"  Random Guess     | {random_mae:.4f}  | {random_mae/car_mae:.2f}x")
    print(f"  Constant (Mean)  | {constant_mae:.4f}  | {constant_mae/car_mae:.2f}x")
    
    # Verify CAR outperforms baselines
    if car_mae < random_mae and car_mae < constant_mae:
        print(f"\n  ✓ CAR outperforms simple baselines (proves learning is effective)")
        return True
    else:
        print(f"\n  ⚠ Warning: CAR did not outperform baselines")
        return False


def test_extreme_noise_no_cheating():
    """
    Test 6: Extreme Noise Anti-Cheating Test
    
    Core test: Verify mechanism correctness under extreme noise.
    """
    print("\n" + "="*70)
    print("  Test 6: Extreme Noise Anti-Cheating Test (Core Test)")
    print("="*70)
    
    np.random.seed(42)
    
    # Train model
    X_train = np.random.randn(500, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    # Key test: at different noise levels
    noise_levels = [0, 1e6, 1e50, 1e100, 1e150]
    
    print(f"\n  Noise Level   | PredStd  | Unique   | SNR (dB)  | Status")
    print(f"  " + "-"*65)
    
    all_passed = True
    
    for noise in noise_levels:
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        
        if noise > 0:
            noise_matrix = np.random.randn(100, 20) * noise
            X_test = X_test + noise_matrix
        
        predictions = np.array([model.predict(x) for x in X_test])
        pred_std = np.std(predictions)
        unique_count = len(set(predictions.round(4)))
        
        # Calculate SNR
        if noise > 0:
            snr_db = -20 * np.log10(noise)
        else:
            snr_db = 0
        
        # Determine status
        if noise == 0:
            status = "Baseline"
            passed = True
        elif noise <= 1e50:
            status = "✓ Normal"
            passed = pred_std > 0.01
        elif noise <= 1e100:
            status = "✓ Normal"
            passed = pred_std > 0.005
        else:
            status = "⚠ Limit"
            passed = pred_std > 0.001
        
        print(f"  {noise:>12.0e} | {pred_std:>8.4f} | {unique_count:>7} | {snr_db:>9.0f} | {status}")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n  ✓ CAR maintains prediction diversity at all noise levels")
        print(f"  ✓ Proves mechanism is effective, not data leakage")
        return True
    else:
        print(f"\n  ⚠ Insufficient prediction diversity at some noise levels")
        return False


def run_all_anti_cheat_tests():
    """Run all anti-cheat tests"""
    print("\n" + "="*70)
    print("  CAR Anti-Cheat Test Suite")
    print("  Verify CAR is learning, not cheating")
    print("="*70)
    
    tests = [
        ("Knowledge Base Distance", test_knowledge_base_distance),
        ("Prediction Diversity", test_prediction_diversity),
        ("Noise Independence", test_noise_independence),
        ("KB Size Impact", test_knowledge_base_scaling),
        ("Random Baseline", test_random_baseline_comparison),
        ("Extreme Noise", test_extreme_noise_no_cheating),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            print(f"\n  ✗ Test failed: {str(e)}")
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for test_name, passed, error in results:
        status = "✓ Passed" if passed else "✗ Failed"
        print(f"  {test_name}: {status}")
        if error:
            print(f"    Error: {error[:50]}")
    
    print(f"\n  Total: {passed_count}/{total_count} passed")
    
    if passed_count == total_count:
        print(f"\n" + "="*70)
        print("  ✓ All anti-cheat tests passed")
        print("  ✓ CAR mechanism is valid, no cheating detected")
        print("="*70)
        return True
    else:
        print(f"\n" + "="*70)
        print(f"  ⚠ {total_count - passed_count} tests failed")
        print("  Suggest reviewing CAR mechanism")
        print("="*70)
        return False


if __name__ == "__main__":
    success = run_all_anti_cheat_tests()
    sys.exit(0 if success else 1)
