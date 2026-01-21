#!/usr/bin/env python3
"""
CAR State Visualization Script
==============================
Generate visualization charts for A (Activation), V (Validation), and X (Feature) states

Usage:
    python visualize_car.py                    # Run full visualization
    python visualize_car.py --noise 1e50       # Test extreme noise
    python visualize_car.py --save             # Save charts

Date: January 2026
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CARConfig
from src.car_model import CompleteCARModel


def setup_matplotlib():
    """Configure matplotlib for proper display"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100


def plot_A_states(model: CompleteCARModel, ax: plt.Axes = None):
    """
    Plot the distribution of Activation state A
    
    A (Activation) represents the current activation level of each computational unit
    """
    states = model.get_all_states()
    A = states['A']
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Histogram
    ax.hist(A, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(A), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(A):.3f}')
    ax.axvline(np.median(A), color='green', linestyle='-.', linewidth=2,
               label=f'Median: {np.median(A):.3f}')
    
    ax.set_xlabel('Activation (A)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Unit Activation States (A)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_V_states(model: CompleteCARModel, ax: plt.Axes = None):
    """
    Plot the distribution of Validation state V
    
    V (Validation) represents the historical validation score of each computational unit
    """
    states = model.get_all_states()
    v = states['v']
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Histogram
    ax.hist(v, bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(np.mean(v), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(v):.3f}')
    ax.axvline(np.median(v), color='green', linestyle='-.', linewidth=2,
               label=f'Median: {np.median(v):.3f}')
    
    ax.set_xlabel('Validation (v)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Unit Validation States (v)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_X_statistics(model: CompleteCARModel, ax: plt.Axes = None):
    """
    Plot Feature statistics X
    
    X represents the statistical information of feature vectors stored in the knowledge base
    """
    states = model.get_all_states()
    X = states['X']
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Bar chart showing feature statistics
    categories = ['Mean', 'Std', 'Min', 'Max']
    values = [X['feature_mean'], X['feature_std'], 
              X['feature_min'], X['feature_max']]
    
    colors = ['steelblue', 'coral', 'seagreen', 'gold']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Feature Statistics (X) - {X["n_patterns"]} Patterns', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    return ax


def plot_AV_correlation(model: CompleteCARModel, ax: plt.Axes = None):
    """
    Plot the correlation between A and V states
    
    Show the relationship between activation state and validation state
    """
    states = model.get_all_states()
    A = np.array(states['A'])
    v = np.array(states['v'])
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(A, v, alpha=0.6, c='steelblue', s=50, edgecolors='black')
    
    # Add trend line
    z = np.polyfit(A, v, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(A), max(A), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend: v = {z[0]:.3f}A + {z[1]:.3f}')
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(A, v)[0, 1]
    
    ax.set_xlabel('Activation (A)', fontsize=12)
    ax.set_ylabel('Validation (v)', fontsize=12)
    ax.set_title(f'Correlation between A and v (r = {correlation:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_unit_states_heatmap(model: CompleteCARModel, ax: plt.Axes = None):
    """
    Plot heatmap of all unit states
    
    Show A and V states for 50 computational units
    """
    states = model.get_all_states()
    A = np.array(states['A'])
    v = np.array(states['v'])
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Create data matrix
    data = np.vstack([A, v])
    
    # Heatmap
    im = ax.imshow(data, aspect='auto', cmap='RdYlBu', interpolation='nearest')
    
    ax.set_ylabel('State Type', fontsize=12)
    ax.set_xlabel('Unit ID', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Activation (A)', 'Validation (v)'])
    ax.set_title('Unit States Heatmap (50 Units)', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='State Value')
    
    return ax


def plot_noise_comparison(models: Dict[str, CompleteCARModel], ax: plt.Axes = None):
    """
    Compare prediction standard deviation at different noise levels
    
    Show CAR robustness under extreme noise
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    noise_levels = []
    pred_stds = []
    labels = []
    
    for label, model in models.items():
        stats = model.get_prediction_statistics()
        noise_levels.append(label)
        pred_stds.append(stats['std'])
        labels.append(label)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_levels)))
    bars = ax.bar(noise_levels, pred_stds, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Prediction Std', fontsize=12)
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_title('Prediction Diversity at Different Noise Levels', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_comprehensive_analysis(model: CompleteCARModel, noise_levels: List[float]):
    """
    Generate complete CAR state analysis charts
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 2x2 subplot layout
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # 1. A state distribution
    plot_A_states(model, ax1)
    
    # 2. V state distribution
    plot_V_states(model, ax2)
    
    # 3. X feature statistics
    plot_X_statistics(model, ax3)
    
    # 4. A-V correlation
    plot_AV_correlation(model, ax4)
    
    plt.tight_layout()
    
    return fig


def run_visualization_demo():
    """Run full visualization demo"""
    print("="*70)
    print("  CAR State Visualization Demo")
    print("="*70)
    
    setup_matplotlib()
    
    # Train model
    np.random.seed(42)
    print("\n1. Training CAR model...")
    X_train = np.random.randn(300, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    kb_stats = model.get_knowledge_base_stats()
    print(f"   Knowledge base size: {kb_stats['size']}")
    print(f"   ✓ Training completed")
    
    # Test different noise levels
    print("\n2. Testing different noise levels...")
    test_noises = [1, 1e6, 1e50, 1e100, 1e150]
    models_for_comparison = {}
    
    for noise in test_noises:
        np.random.seed(123)
        X_test = np.random.randn(100, 20)
        noise_matrix = np.random.randn(100, 20) * noise
        X_noisy = X_test + noise_matrix
        
        # Create new model for testing
        test_model = CompleteCARModel(config=config, n_features=20)
        test_model.fit(X_train, y_train)
        
        predictions = [test_model.predict(x) for x in X_noisy]
        pred_std = np.std(predictions)
        unique_count = len(set(predictions.round(4)))
        
        snr_db = -20 * np.log10(noise) if noise > 0 else 0
        status = "✓ Normal" if pred_std > 0.01 else "✗ Collapsed"
        
        print(f"   Noise {noise:>10.0e} | SNR: {snr_db:>7.0f} dB | "
              f"Std: {pred_std:.4f} | {status}")
        
        models_for_comparison[f'{noise:.0e}'] = test_model
    
    # Generate visualizations
    print("\n3. Generating visualization charts...")
    
    # Comprehensive analysis chart
    fig = plot_comprehensive_analysis(model)
    fig.suptitle('CAR Complete State Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('figures/car_state_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: figures/car_state_analysis.png")
    
    # A-V correlation chart
    fig2, ax = plt.subplots(figsize=(10, 8))
    plot_AV_correlation(model, ax)
    plt.savefig('figures/car_AV_correlation.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: figures/car_AV_correlation.png")
    
    # Unit states heatmap
    fig3, ax = plt.subplots(figsize=(12, 6))
    plot_unit_states_heatmap(model, ax)
    plt.savefig('figures/car_unit_heatmap.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: figures/car_unit_heatmap.png")
    
    # Noise comparison chart
    fig4, ax = plt.subplots(figsize=(10, 6))
    plot_noise_comparison(models_for_comparison, ax)
    plt.savefig('figures/car_noise_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: figures/car_noise_comparison.png")
    
    # Print final statistics
    print("\n4. Final state statistics:")
    states = model.get_all_states()
    unit_stats = model.get_unit_statistics()
    
    print(f"   A (Activation) states:")
    print(f"     - Mean: {np.mean(states['A']):.4f}")
    print(f"     - Std: {np.std(states['A']):.4f}")
    print(f"     - Range: [{min(states['A']):.4f}, {max(states['A']):.4f}]")
    
    print(f"\n   V (Validation) states:")
    print(f"     - Mean: {np.mean(states['v']):.4f}")
    print(f"     - Std: {np.std(states['v']):.4f}")
    print(f"     - Range: [{min(states['v']):.4f}, {max(states['v']):.4f}]")
    
    print(f"\n   X (Feature) statistics:")
    print(f"     - Number of patterns: {states['X']['n_patterns']}")
    print(f"     - Feature dimension: {states['X']['feature_dim']}")
    print(f"     - Feature mean: {states['X']['feature_mean']:.4f}")
    print(f"     - Feature std: {states['X']['feature_std']:.4f}")
    
    print("\n" + "="*70)
    print("  Visualization completed!")
    print("  Charts saved to figures/ directory")
    print("="*70)
    
    return model


def test_specific_noise(noise_level: float):
    """Test specific noise level and generate visualization"""
    setup_matplotlib()
    
    # Train
    np.random.seed(42)
    X_train = np.random.randn(300, 20)
    y_train = np.sum(np.sin(X_train[:, :3]), axis=1) + np.cos(X_train[:, 4])
    
    config = CARConfig(KB_CAPACITY=100)
    model = CompleteCARModel(config=config, n_features=20)
    model.fit(X_train, y_train)
    
    # Test noise
    np.random.seed(123)
    X_test = np.random.randn(100, 20)
    noise = np.random.randn(100, 20) * noise_level
    X_noisy = X_test + noise
    
    predictions = [model.predict(x) for x in X_noisy]
    
    # Generate visualization
    fig = plot_comprehensive_analysis(model)
    snr_db = -20 * np.log10(noise_level)
    fig.suptitle(f'CAR State Analysis at Noise = {noise_level:.0e} (SNR = {snr_db:.0f} dB)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(f'figures/car_noise_{noise_level:.0e}.png', dpi=150, bbox_inches='tight')
    
    print(f"Noise {noise_level:.0e}: PredStd = {np.std(predictions):.4f}")
    print(f"Chart saved: figures/car_noise_{noise_level:.0e}.png")
    
    return model


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--noise":
            noise = float(sys.argv[2]) if len(sys.argv) > 2 else 1e50
            test_specific_noise(noise)
        else:
            print("Usage: python visualize_car.py [--noise <level>]")
    else:
        run_visualization_demo()
