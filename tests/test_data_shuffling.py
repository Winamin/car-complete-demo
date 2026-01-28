#!/usr/bin/env python3
# CAR Data Shuffling Test - Verify Learning Ability vs Memory Ability

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CARConfig
from src.car_model import CompleteCARModel


def generate_data(n_samples=1000, n_features=20, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.sin(X[:, 3])
    return X, y


def test_original_data():
    print("\nTest 1: Original Data")
    print("-" * 60)
    
    X_train, y_train = generate_data(500, 20, 42)
    X_test, y_test = generate_data(200, 20, 123)
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    predictions = np.array([model.predict(x) for x in X_test])
    mse = np.mean((predictions - y_test) ** 2)
    
    print(f"Test set MSE: {mse:.4f}")
    print(f"Knowledge base size: {len(model.knowledge_base.patterns)}")
    
    return mse


def test_shuffled_correspondence():
    print("\nTest 2: Shuffled Correspondence")
    print("-" * 60)
    
    X_train, y_train = generate_data(500, 20, 42)
    X_test, y_test = generate_data(200, 20, 123)
    
    np.random.shuffle(y_train)
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    predictions = np.array([model.predict(x) for x in X_test])
    mse = np.mean((predictions - y_test) ** 2)
    
    print(f"Test set MSE: {mse:.4f}")
    print(f"Knowledge base size: {len(model.knowledge_base.patterns)}")
    
    return mse


def test_random_correspondence():
    print("\nTest 3: Completely Random Correspondence")
    print("-" * 60)
    
    X_train, y_train = generate_data(500, 20, 42)
    X_test, y_test = generate_data(200, 20, 123)
    
    y_random = np.random.randn(len(y_train))
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_random)
    
    predictions = np.array([model.predict(x) for x in X_test])
    mse = np.mean((predictions - y_test) ** 2)
    
    print(f"Test set MSE: {mse:.4f}")
    print(f"Knowledge base size: {len(model.knowledge_base.patterns)}")
    
    return mse


def run_data_shuffling_test():
    print("=" * 60)
    print("  CAR Data Shuffling Test")
    print("=" * 60)
    print("\nGoal: Verify CAR learns patterns instead of memorizing data")
    
    mse_original = test_original_data()
    mse_shuffled = test_shuffled_correspondence()
    mse_random = test_random_correspondence()
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    print(f"Original data MSE: {mse_original:.4f}")
    print(f"Shuffled correspondence MSE: {mse_shuffled:.4f}")
    print(f"Random correspondence MSE: {mse_random:.4f}")
    
    ratio = mse_shuffled / mse_original
    print(f"\nPerformance degradation ratio: {ratio:.2f}x")
    
    if ratio > 2.0:
        print("✓ CAR learned patterns, not memorization")
    elif ratio > 1.5:
        print("⚠ CAR partially learned patterns")
    else:
        print("✗ CAR may have over-memorized data")
    
    return {
        'original_mse': mse_original,
        'shuffled_mse': mse_shuffled,
        'random_mse': mse_random,
        'performance_ratio': ratio
    }


if __name__ == "__main__":
    results = run_data_shuffling_test()
    sys.exit(0)
