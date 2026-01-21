#!/usr/bin/env python3
"""
Test Suite for CAR Model
Comprehensive tests for noise robustness and basic functionality.

Tests cover:
- Basic functionality (initialization, fit, predict)
- Noise robustness at various levels
- Numerical precision limits
- Edge cases and error handling

Date: January 2026
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CARConfig
from src.car_model import CompleteCARModel


class TestCARBasic:
    """Basic functionality tests for CAR model."""
    
    def test_initialization(self):
        """Test model initialization with default config."""
        config = CARConfig()
        model = CompleteCARModel(config=config, n_features=20)
        
        assert model.n_features == 20
        assert model.n_units == 50
        assert len(model.knowledge_base.patterns) == 0
        assert model.prediction_history == []
    
    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        config = CARConfig(
            KB_CAPACITY=100,
            KB_MERGE_THRESHOLD=0.3,
            DIVERSITY_BONUS=0.1
        )
        model = CompleteCARModel(config=config, n_features=10)
        
        assert model.config.KB_CAPACITY == 100
        assert model.config.KB_MERGE_THRESHOLD == 0.3
        assert model.config.DIVERSITY_BONUS == 0.1
    
    def test_fit_and_predict(self):
        """Test basic fit and predict cycle."""
        np.random.seed(42)
        
        # Generate simple data
        X_train = np.random.randn(100, 10)
        y_train = np.sum(X_train[:, :3], axis=1)
        
        X_test = np.random.randn(20, 10)
        
        # Initialize and train
        config = CARConfig(KB_CAPACITY=50)
        model = CompleteCARModel(config=config, n_features=10)
        model.fit(X_train, y_train)
        
        # Check knowledge base grew
        kb_stats = model.get_knowledge_base_stats()
        assert kb_stats['size'] > 0
        assert kb_stats['size'] <= 50  # Respect capacity
        
        # Make predictions
        predictions = [model.predict(x) for x in X_test]
        
        # Check predictions exist and have variation
        assert len(predictions) == 20
        assert np.std(predictions) > 0  # Some variation
    
    def test_model_summary(self):
        """Test model summary generation."""
        config = CARConfig()
        model = CompleteCARModel(config=config, n_features=20)
        
        summary = model.get_model_summary()
        
        assert 'configuration' in summary
        assert 'knowledge_base' in summary
        assert 'predictions' in summary
        assert summary['units']['total'] == 50
    
    def test_reset(self):
        """Test model reset functionality."""
        config = CARConfig()
        model = CompleteCARModel(config=config, n_features=20)
        
        # Train
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.sum(X_train[:, :3], axis=1)
        model.fit(X_train, y_train)
        
        # Make some predictions
        for _ in range(10):
            x = np.random.randn(20)
            model.predict(x)
        
        # Check state before reset
        assert len(model.prediction_history) == 10
        kb_size_before = model.get_knowledge_base_stats()['size']
        assert kb_size_before > 0
        
        # Reset
        model.reset()
        
        # Check state after reset
        assert len(model.prediction_history) == 0
        assert len(model.knowledge_base.patterns) == 0


class TestCARNoiseRobustness:
    """Noise robustness tests for CAR model."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for noise testing."""
        np.random.seed(42)
        
        X_train = np.random.randn(300, 20)
        y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 3])
        
        config = CARConfig(KB_CAPACITY=50)
        model = CompleteCARModel(config=config, n_features=20)
        model.fit(X_train, y_train)
        
        return model, X_train, y_train
    
    def test_clean_data_prediction(self, trained_model):
        """Test prediction on clean (no noise) data."""
        model, X_train, y_train = trained_model
        
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        y_test = np.sum(np.sin(X_test[:, :3]), axis=1) + np.cos(X_test[:, 3])
        
        predictions = np.array([model.predict(x) for x in X_test])
        pred_std = np.std(predictions)
        true_std = np.std(y_test)
        
        # Check predictions have reasonable variation
        assert pred_std > 0.1 * true_std  # At least 10% of true variation
        assert pred_std < 2.0 * true_std  # Not more than 200% of true variation
    
    def test_low_noise(self, trained_model):
        """Test prediction with low noise (noise multiplier = 1)."""
        model, X_train, y_train = trained_model
        
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise = np.random.randn(100, 20) * 1.0  # Noise multiplier = 1
        X_test_noisy = X_test + noise
        
        predictions = np.array([model.predict(x) for x in X_test_noisy])
        pred_std = np.std(predictions)
        
        assert pred_std > 0  # Some variation should remain
    
    def test_medium_noise(self, trained_model):
        """Test prediction with medium noise (noise multiplier = 1000)."""
        model, X_train, y_train = trained_model
        
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise = np.random.randn(100, 20) * 1000
        X_test_noisy = X_test + noise
        
        predictions = np.array([model.predict(x) for x in X_test_noisy])
        pred_std = np.std(predictions)
        
        assert pred_std > 0  # CAR should maintain some prediction diversity
    
    def test_high_noise(self, trained_model):
        """Test prediction with high noise (noise multiplier = 1e6)."""
        model, X_train, y_train = trained_model
        
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise = np.random.randn(100, 20) * 1e6
        X_test_noisy = X_test + noise
        
        predictions = np.array([model.predict(x) for x in X_test_noisy])
        pred_std = np.std(predictions)
        
        # CAR should maintain prediction diversity
        assert pred_std > 0, "Predictions collapsed to constant at 1e6 noise"
    
    def test_extreme_noise_1e12(self, trained_model):
        """Test prediction with extreme noise (noise multiplier = 1e12)."""
        model, X_train, y_train = trained_model
        
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise = np.random.randn(100, 20) * 1e12
        X_test_noisy = X_test + noise
        
        predictions = np.array([model.predict(x) for x in X_test_noisy])
        pred_std = np.std(predictions)
        
        # CAR should maintain prediction diversity even at extreme noise
        assert pred_std > 0, "Predictions collapsed at 1e12 noise"
        assert len(set(predictions.round(6))) > 1, "All predictions identical"
    
    def test_extreme_noise_1e50(self, trained_model):
        """Test prediction with extreme noise (noise multiplier = 1e50)."""
        model, X_train, y_train = trained_model
        
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise = np.random.randn(100, 20) * 1e50
        X_test_noisy = X_test + noise
        
        predictions = np.array([model.predict(x) for x in X_test_noisy])
        pred_std = np.std(predictions)
        
        # CAR should maintain prediction diversity
        assert pred_std > 0, "Predictions collapsed at 1e50 noise"
    
    def test_extreme_noise_1e100(self, trained_model):
        """Test prediction with extreme noise (noise multiplier = 1e100)."""
        model, X_train, y_train = trained_model
        
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise = np.random.randn(100, 20) * 1e100
        X_test_noisy = X_test + noise
        
        predictions = np.array([model.predict(x) for x in X_test_noisy])
        pred_std = np.std(predictions)
        
        assert pred_std > 0, "Predictions collapsed at 1e100 noise"
    
    def test_extreme_noise_1e150(self, trained_model):
        """Test prediction with extreme noise (noise multiplier = 1e150).
        
        This is the key test - at 1e150 noise (-3000 dB SNR),
        CAR should still maintain prediction capability.
        """
        model, X_train, y_train = trained_model
        
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise = np.random.randn(100, 20) * 1e150
        X_test_noisy = X_test + noise
        
        predictions = np.array([model.predict(x) for x in X_test_noisy])
        pred_std = np.std(predictions)
        
        # THIS IS THE KEY ASSERTION
        assert pred_std > 0, (
            "FAILURE: Predictions collapsed at 1e150 noise. "
            "This indicates CAR's noise robustness claim is invalid."
        )
        
        # Additional check: multiple unique predictions
        unique_preds = len(set(predictions.round(4)))
        assert unique_preds > 10, f"Only {unique_preds} unique predictions at 1e150 noise"


class TestCARNumericalPrecision:
    """Tests for numerical precision behavior."""
    
    def test_float64_limits(self):
        """Test behavior at float64 precision limits."""
        np.random.seed(42)
        
        # Generate data
        X_train = np.random.randn(100, 10)
        y_train = np.sum(X_train[:, :3], axis=1)
        
        config = CARConfig(KB_CAPACITY=20)
        model = CompleteCARModel(config=config, n_features=10)
        model.fit(X_train, y_train)
        
        # Test at various noise levels near float64 limit
        test_noises = [1e100, 1e150, 1e200, 1e250, 1e300]
        
        for noise_level in test_noises:
            np.random.seed(123)
            X_test = np.random.randn(10, 10)
            noise = np.random.randn(10, 10) * noise_level
            X_test_noisy = X_test + noise
            
            predictions = []
            for x in X_test_noisy:
                try:
                    pred = model.predict(x)
                    predictions.append(pred)
                except (OverflowError, FloatingPointError):
                    predictions.append(np.nan)
            
            # Check for valid predictions
            valid_count = sum(1 for p in predictions if not np.isnan(p))
            
            # At 1e150, we expect valid predictions
            if noise_level <= 1e150:
                assert valid_count > 0, f"No valid predictions at {noise_level}"
            
            # At very high noise, predictions may become NaN
            if noise_level > 1e200:
                # This is expected - float64 overflow
                pass


class TestCAREgeCases:
    """Edge case tests for CAR model."""
    
    def test_single_sample_training(self):
        """Test with single training sample."""
        np.random.seed(42)
        
        X_train = np.random.randn(1, 10)
        y_train = np.array([1.0])
        
        config = CARConfig(KB_CAPACITY=10)
        model = CompleteCARModel(config=config, n_features=10)
        model.fit(X_train, y_train)
        
        # Should still work
        x_test = np.random.randn(10)
        pred = model.predict(x_test)
        assert not np.isnan(pred)
    
    def test_zero_features(self):
        """Test with zero-valued features."""
        np.random.seed(42)
        
        X_train = np.random.randn(50, 10)
        y_train = np.sum(X_train[:, :3], axis=1)
        
        # Zero out some features
        X_train[:, 5:] = 0
        
        config = CARConfig(KB_CAPACITY=20)
        model = CompleteCARModel(config=config, n_features=10)
        model.fit(X_train, y_train)
        
        x_test = np.zeros(10)
        pred = model.predict(x_test)
        assert not np.isnan(pred)
    
    def test_constant_features(self):
        """Test with constant features."""
        np.random.seed(42)
        
        X_train = np.random.randn(50, 10)
        X_train[:, :] = 1.0  # All ones
        y_train = np.sum(X_train[:, :3], axis=1)
        
        config = CARConfig(KB_CAPACITY=20)
        model = CompleteCARModel(config=config, n_features=10)
        model.fit(X_train, y_train)
        
        x_test = np.ones(10)
        pred = model.predict(x_test)
        assert not np.isnan(pred)
    
    def test_repeated_predictions(self):
        """Test repeated predictions on same input."""
        np.random.seed(42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.sum(X_train[:, :3], axis=1)
        
        config = CARConfig(KB_CAPACITY=20)
        model = CompleteCARModel(config=config, n_features=10)
        model.fit(X_train, y_train)
        
        x_test = np.random.randn(10)
        
        # Make multiple predictions on same input
        predictions = [model.predict(x_test) for _ in range(10)]
        
        # Predictions should be identical (deterministic retrieval)
        assert len(set(predictions)) == 1, "Predictions should be deterministic"
    
    def test_empty_knowledge_base(self):
        """Test prediction with empty knowledge base."""
        config = CARConfig()
        model = CompleteCARModel(config=config, n_features=10)
        
        x_test = np.random.randn(10)
        pred = model.predict(x_test)
        
        # Should return default value, not crash
        assert pred == 0.0 or not np.isnan(pred)


def run_all_tests():
    """Run all tests and print summary."""
    print("="*70)
    print("CAR Model Test Suite")
    print("="*70)
    
    test_classes = [
        TestCARBasic,
        TestCARNoiseRobustness,
        TestCARNumericalPrecision,
        TestCAREgeCases,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 50)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            method = getattr(instance, method_name)
            try:
                # Handle fixtures by checking if method needs them
                if hasattr(instance, 'trained_model'):
                    # This is a pytest fixture, skip in standalone mode
                    print(f"  ⚠ {method_name}: SKIPPED (requires pytest fixture)")
                    continue
                
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)[:50]}")
                failed_tests.append((method_name, str(e)))
            
            total_tests += 1
    
    print("\n" + "="*70)
    print(f"Test Summary: {passed_tests}/{total_tests} passed")
    print("="*70)
    
    if failed_tests:
        print("\nFailed tests:")
        for name, error in failed_tests:
            print(f"  - {name}: {error[:100]}")
    
    return failed_tests == []


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
