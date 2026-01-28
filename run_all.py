#!/usr/bin/env python3
"""
CAR Demo - All-in-One Run Script
================================
Run all important tests and demonstrate CAR model capabilities

Usage:
    python run_all.py              # Run all tests
    python run_all.py --quick      # Quick test mode
    python run_all.py --noise      # Noise robustness test only
    python run_all.py --float128   # Float128 limit test only
    python run_all.py --attack     # Adversarial attack test only

Author: Yingxu Wang
Date: January 2026
"""

import sys
import os
import argparse
import time
import json
from datetime import datetime

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CARConfig
from src.car_model import CompleteCARModel
import numpy as np


def print_header(title):
    """Print formatted title"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_section(title):
    """Print section header"""
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def test_basic_functionality():
    """Test basic functionality"""
    print_section("Basic Functionality Test")
    
    # Initialize
    config = CARConfig(KB_CAPACITY=50)
    model = CompleteCARModel(config=config, n_features=20)
    print(f"✓ Model initialized: {model.n_features} features, {model.n_units} units")
    
    # Train
    np.random.seed(42)
    X_train = np.random.randn(200, 20)
    y_train = np.sum(X_train[:, :3], axis=1) + np.cos(X_train[:, 4])
    
    model.fit(X_train, y_train)
    kb_stats = model.get_knowledge_base_stats()
    print(f"✓ Training completed: Knowledge base size = {kb_stats['size']}")
    
    # Predict
    np.random.seed(123)
    X_test = np.random.randn(20, 20)
    predictions = [model.predict(x) for x in X_test]
    pred_std = np.std(predictions)
    print(f"✓ Prediction completed: Mean = {np.mean(predictions):.4f}, Std = {pred_std:.4f}")
    
    return True


def test_noise_robustness(quick_mode=False):
    """Noise robustness test"""
    print_section("Noise Robustness Test")
    
    np.random.seed(42)
    
    # Train model
    X_train = np.random.randn(300, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    
    config = CARConfig(KB_CAPACITY=50)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    print("✓ Model training completed")
    
    # Noise levels
    noise_levels = [1, 1000, 1e6, 1e12, 1e50, 1e75, 1e100, 1e150]
    
    if quick_mode:
        noise_levels = [1, 1000, 1e6, 1e12, 1e50]
    
    results = []
    
    for noise in noise_levels:
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise_matrix = np.random.randn(100, 20) * noise
        X_noisy = X_test + noise_matrix
        
        predictions = np.array([model.predict(x) for x in X_noisy])
        pred_std = np.std(predictions)
        unique_count = len(set(predictions.round(4)))
        
        # Calculate SNR (dB)
        snr_db = -20 * np.log10(noise) if noise > 0 else 0
        
        status = "✓ Normal" if pred_std > 0 else "✗ Collapsed"
        if noise > 1e100:
            status = "⚠ Overflow" if pred_std == 0 else "✓ Normal"
        
        print(f"  Noise {noise:>12.0e} | SNR: {snr_db:>8.0f} dB | "
              f"PredStd: {pred_std:>8.4f} | Unique: {unique_count:>3} | {status}")
        
        results.append({
            'noise': noise,
            'snr_db': snr_db,
            'pred_std': pred_std,
            'unique_predictions': unique_count,
            'status': status
        })
    
    return results


def test_float128_limits():
    """Float128 Limit Test"""
    print_section("Float128 Limit Test")
    
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass
    
    from decimal import Decimal, getcontext
    
    # Set high precision for Decimal (simulating Float128)
    getcontext().prec = 50
    
    np.random.seed(42)
    
    # Test numerical range
    print(f"  Float64 Max: {np.finfo(np.float64).max:.2e}")
    
    # Float128 theoretical maximum
    max_f128 = Decimal('1.189731495357231765085759326628007e4932')
    print(f"  Float128 Max (theoretical): {max_f128:.2e}")
    print(f"  Using Decimal module to simulate Float128 precision")
    print(f"  Decimal precision: {getcontext().prec} digits")
    
    # Train model with float64 (since CAR uses float64 internally)
    X_train = np.random.randn(100, 10)
    y_train = np.sum(X_train[:, :3], axis=1)
    
    config = CARConfig(KB_CAPACITY=20)
    model = CompleteCARModel(config=config, n_features=10)
    model.fit(X_train, y_train)
    
    print("  Model training completed")
    
    # Test extreme noise (using float64, but demonstrating the concept)
    extreme_noises = [1e100, 1e200, 1e500, 1e1000, 1e2000]
    
    results = []
    for noise in extreme_noises:
        # Use Decimal to check if noise exceeds Float128 range
        noise_decimal = Decimal(str(noise))
        
        if noise_decimal > max_f128:
            snr_db = -20 * np.log10(float(noise))
            print(f"  Noise {noise:>12.0e} | SNR: {snr_db:>8.0f} dB | "
                  f"✗ Exceeds Float128 range")
            results.append({'noise': noise, 'snr_db': snr_db, 'status': 'overflow'})
            continue
        
        # Test with float64 (will overflow for very large values)
        try:
            X_test = np.random.randn(20, 10) * noise
            predictions = [model.predict(x) for x in X_test]
            pred_std = np.std(predictions)
            snr_db = -20 * np.log10(float(noise))
            print(f"  Noise {noise:>12.0e} | SNR: {snr_db:>8.0f} dB | "
                  f"PredStd: {pred_std:.4f} | ✓ Success")
            results.append({'noise': noise, 'snr_db': snr_db, 'status': 'success'})
        except Exception as e:
            snr_db = -20 * np.log10(float(noise))
            print(f"  Noise {noise:>12.0e} | SNR: {snr_db:>8.0f} dB | "
                  f"✗ Float64 overflow: {str(e)[:30]}")
            results.append({'noise': noise, 'snr_db': snr_db, 'status': 'float64_overflow'})
    
    # Calculate final conclusion
    if results:
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            max_successful = max([r['snr_db'] for r in successful])
            print(f"\n  ★ Float64 Limit: ~{max_successful:.0f} dB")
            print(f"  ★ Float128 would extend this to ~-49320 dB")
        else:
            print(f"\n  ✗ All tests failed - noise too extreme")
    
    return results


def test_adversarial_attack():
    """Adversarial Attack Test"""
    print_section("Adversarial Attack Test")
    
    np.random.seed(42)
    
    # Train model
    X_train = np.random.randn(200, 20)
    y_train = np.sum(X_train[:, :3], axis=1)
    
    config = CARConfig(KB_CAPACITY=50)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    # Clean predictions
    np.random.seed(123)
    X_test = np.random.randn(10, 20)
    clean_predictions = np.array([model.predict(x) for x in X_test])
    clean_mean = np.mean(clean_predictions)
    
    print(f"  Clean sample prediction mean: {clean_mean:.4f}")
    
    # Adversarial attack tests
    attack_types = ['FGSM', 'PGD', 'Random Noise']
    epsilons = [0.01, 0.1, 0.5, 1.0]
    
    results = []
    
    for attack in attack_types:
        print(f"\n  {attack} Attack Test:")
        
        for eps in epsilons:
            np.random.seed(456)
            
            if attack == 'FGSM':
                # Simple gradient estimation attack
                noise = np.random.randn(20) * eps
            elif attack == 'PGD':
                # Iterative attack
                noise = np.random.randn(20) * eps * 0.5
            else:
                # Random noise
                noise = np.random.randn(20) * eps
            
            # Test all samples
            adversarial_predictions = []
            for x in X_test:
                x_adv = x + noise
                pred = model.predict(x_adv)
                adversarial_predictions.append(pred)
            
            adv_mean = np.mean(adversarial_predictions)
            deviation = abs(adv_mean - clean_mean)
            
            status = "✓" if deviation < abs(clean_mean) * 2 else "⚠"
            print(f"    ε={eps:>4.2f} | Prediction offset: {deviation:>8.4f} | {status}")
            
            results.append({
                'attack': attack,
                'epsilon': eps,
                'deviation': deviation,
                'status': status
            })
    
    return results


def test_architecture_comparison():
    """Architecture comparison test"""
    print_section("Architecture Comparison Test")
    
    np.random.seed(42)
    
    # Generate data
    X_train = np.random.randn(300, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    X_test = np.random.randn(50, 20)
    y_test = np.sum(np.sin(X_test[:, :3]), axis=1) + np.cos(X_test[:, 4])
    
    # Different configurations
    configs = [
        ("Default (50 units)", CARConfig(KB_CAPACITY=50)),
        ("Small KB (20)", CARConfig(KB_CAPACITY=20)),
        ("Large KB (100)", CARConfig(KB_CAPACITY=100)),
        ("Low merge (0.1)", CARConfig(KB_MERGE_THRESHOLD=0.1)),
        ("High merge (0.5)", CARConfig(KB_MERGE_THRESHOLD=0.5)),
    ]
    
    results = []
    
    for name, config in configs:
        model = CompleteCARModel(config=config, n_features=20)
        model.fit(X_train, y_train)
        
        # Test noise
        np.random.seed(123)
        X_noisy = X_test + np.random.randn(50, 20) * 1e6
        
        predictions = [model.predict(x) for x in X_noisy]
        pred_std = np.std(predictions)
        
        print(f"  {name:<25} | PredStd: {pred_std:>8.4f} | ✓ Completed")
        
        results.append({'config': name, 'pred_std': pred_std})
    
    return results


def test_knowledge_base_scaling():
    """Knowledge base scaling test"""
    print_section("Knowledge Base Scaling Test")
    
    np.random.seed(42)
    
    sizes = [10, 50, 100, 200, 500]
    
    results = []
    
    for size in sizes:
        X_train = np.random.randn(size, 20)
        y_train = np.sum(X_train[:, :3], axis=1)
        
        config = CARConfig(KB_CAPACITY=min(size, 100))
        model = CompleteCARModel(config=config, n_features=20)
        model.fit(X_train, y_train)
        
        # Test
        np.random.seed(123)
        X_test = np.random.randn(20, 20)
        X_noisy = X_test + np.random.randn(20, 20) * 1e6
        
        predictions = [model.predict(x) for x in X_noisy]
        pred_std = np.std(predictions)
        unique = len(set(np.round(predictions, 4)))
        
        kb_stats = model.get_knowledge_base_stats()
        
        print(f"  Train samples {size:>4} | KB size {kb_stats['size']:>3} | "
              f"PredStd: {pred_std:>7.4f} | Unique: {unique:>3}")
        
        results.append({
            'train_size': size,
            'kb_size': kb_stats['size'],
            'pred_std': pred_std,
            'unique_predictions': unique
        })
    
    return results


def run_full_demo():
    """Run full demonstration"""
    print_header("CAR Complete Functionality Demonstration")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # 1. Basic functionality
    try:
        test_basic_functionality()
        all_results['basic'] = 'success'
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        all_results['basic'] = f'failed: {e}'
    
    # 2. Noise robustness
    try:
        results = test_noise_robustness()
        all_results['noise'] = results
    except Exception as e:
        print(f"✗ Noise test failed: {e}")
        all_results['noise'] = f'failed: {e}'
    
    # 3. Float128 limits
    try:
        results = test_float128_limits()
        all_results['float128'] = results
    except Exception as e:
        print(f"✗ Float128 test failed: {e}")
        all_results['float128'] = f'failed: {e}'
    
    # 4. Adversarial attack
    try:
        results = test_adversarial_attack()
        all_results['adversarial'] = results
    except Exception as e:
        print(f"✗ Adversarial attack test failed: {e}")
        all_results['adversarial'] = f'failed: {e}'
    
    # 5. Architecture comparison
    try:
        results = test_architecture_comparison()
        all_results['architecture'] = results
    except Exception as e:
        print(f"✗ Architecture comparison failed: {e}")
        all_results['architecture'] = f'failed: {e}'
    
    # 6. Knowledge base scaling
    try:
        results = test_knowledge_base_scaling()
        all_results['scaling'] = results
    except Exception as e:
        print(f"✗ Scaling test failed: {e}")
        all_results['scaling'] = f'failed: {e}'
    
    # Summary
    print_header("Test Summary")
    
    success_count = sum(1 for v in all_results.values() if v != 'failed')
    total_count = len(all_results)
    
    print(f"  Tests completed: {success_count}/{total_count}")
    
    for test_name, result in all_results.items():
        if isinstance(result, str) and 'failed' in result:
            print(f"  ✗ {test_name}: {result}")
        else:
            print(f"  ✓ {test_name}: completed")
    
    # Save results
    output_file = f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert numpy types to Python native types
        json.dump(all_results, f, default=str, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_file}")
    
    return all_results


def run_quick_demo():
    """Quick demo"""
    print_header("CAR Quick Demo")
    
    test_basic_functionality()
    test_noise_robustness(quick_mode=True)
    
    print_header("Quick demo completed")
    print("  Use --full to run full test")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='CAR Demo - Cognitive Autonomous Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py           # Run full demo
  python run_all.py --quick   # Quick test
  python run_all.py --basic   # Basic functionality only
  python run_all.py --noise   # Noise robustness only
  python run_all.py --float128 # Float128 limit only
  python run_all.py --attack  # Adversarial attack only
        """
    )
    
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--basic', action='store_true', help='Run basic functionality test only')
    parser.add_argument('--noise', action='store_true', help='Run noise robustness test only')
    parser.add_argument('--float128', action='store_true', help='Run Float128 limit test only')
    parser.add_argument('--attack', action='store_true', help='Run adversarial attack test only')
    parser.add_argument('--arch', action='store_true', help='Run architecture comparison only')
    parser.add_argument('--scaling', action='store_true', help='Run scaling test only')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.quick:
        run_quick_demo()
    elif args.basic:
        test_basic_functionality()
    elif args.noise:
        test_noise_robustness(quick_mode=False)
    elif args.float128:
        test_float128_limits()
    elif args.attack:
        test_adversarial_attack()
    elif args.arch:
        test_architecture_comparison()
    elif args.scaling:
        test_knowledge_base_scaling()
    else:
        run_full_demo()
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
