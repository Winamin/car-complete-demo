"""
Example Usage Script for CAR System

This script demonstrates various ways to use the CAR system
for molecular symmetry detection.
"""

import numpy as np
from .enhanced_car import EnhancedCARSystem
from .qm9_dataset import QM9Dataset, MolecularSymmetryGenerator
from .experiment import ExperimentRunner

def example_1_basic_usage():
    """Example 1: Basic usage of CAR system."""
    print("\n" + "="*60)
    print("Example 1: Basic CAR System Usage")
    print("="*60)
    
    # Get QM9 statistics
    qm9 = QM9Dataset()
    stats = qm9.get_qm9_statistics()
    
    # Generate a small dataset
    symmetry_gen = MolecularSymmetryGenerator(stats)
    features, labels = symmetry_gen.generate_complete_dataset(
        n_total=50,
        high_symmetry_ratio=0.25,
        n_atoms=9,
        n_features=5,
        random_seed=42
    )
    
    # Initialize CAR system
    car_system = EnhancedCARSystem(n_units=30, n_chunks=3, random_seed=42)
    
    # Process molecules
    print(f"\nProcessing {len(features)} molecules...")
    correct = 0
    for i, (X, y) in enumerate(zip(features, labels)):
        result = car_system.process_molecule(X, y)
        if result['prediction'] == y:
            correct += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(features)} molecules...")
    
    accuracy = correct / len(features)
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Knowledge base size: {car_system.get_statistics()['knowledge_base_size']}")


def example_2_single_experiment():
    """Example 2: Run a single experiment with detailed analysis."""
    print("\n" + "="*60)
    print("Example 2: Single Experiment with Analysis")
    print("="*60)
    
    # Initialize experiment runner
    runner = ExperimentRunner(n_units=40, n_chunks=4)
    
    # Generate dataset
    qm9 = QM9Dataset()
    stats = qm9.get_qm9_statistics()
    symmetry_gen = MolecularSymmetryGenerator(stats)
    features, labels = symmetry_gen.generate_complete_dataset(
        n_total=500,
        high_symmetry_ratio=0.25,
        n_atoms=9,
        n_features=5,
        random_seed=123
    )
    
    # Run experiment
    results = runner.run_single_experiment(features, labels, random_seed=123)
    
    print(f"\nDetailed Results:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Precision: {results['precision']:.2%}")
    print(f"  Recall: {results['recall']:.2%}")
    print(f"  F1-Score: {results['f1']:.2%}")
    print(f"  Z-Score: {results['z_score']:.2f}")
    print(f"  Cohen's d: {results['cohens_d']:.2f}")
    print(f"  Optimal Threshold: {results['optimal_threshold']:.6f}")


def example_3_multi_experiment():
    """Example 3: Run multiple experiments for statistical validation."""
    print("\n" + "="*60)
    print("Example 3: Multi-Experiment Validation")
    print("="*60)
    
    # Initialize experiment runner
    runner = ExperimentRunner(n_units=30, n_chunks=3)
    
    # Run 3 experiments (reduced from 10 for demonstration)
    results = runner.run_multi_experiment(
        n_experiments=3,
        n_molecules=500,
        high_symmetry_ratio=0.25,
        n_atoms=9,
        n_features=5,
        base_seed=456
    )
    
    print(f"\nAggregated Results:")
    print(f"  Mean Accuracy: {results['mean_accuracy']:.2%} ± {results['std_accuracy']:.2%}")
    print(f"  Mean F1-Score: {results['mean_f1']:.2%} ± {results['std_f1']:.2%}")
    print(f"  Mean Precision: {results['mean_precision']:.2%} ± {results['std_precision']:.2%}")
    print(f"  Mean Recall: {results['mean_recall']:.2%} ± {results['std_recall']:.2%}")
    print(f"  Mean Z-Score: {results['mean_z_score']:.2f} ± {results['std_z_score']:.2f}")
    print(f"  Total Time: {results['total_time']:.2f} seconds")


def example_4_custom_molecules():
    """Example 4: Process custom molecular features."""
    print("\n" + "="*60)
    print("Example 4: Custom Molecular Features")
    print("="*60)
    
    # Initialize CAR system
    car_system = EnhancedCARSystem(n_units=20, n_chunks=2, random_seed=789)
    
    # Create custom high-symmetry molecule (water-like C2v)
    print("\nCreating custom high-symmetry molecule (C2v)...")
    n_atoms = 9
    n_features = 5
    high_sym_X = np.random.normal(0.25, 0.02, (n_atoms, n_features))
    # Apply C2v symmetry
    for i in range(n_atoms // 2):
        high_sym_X[2*i+1, :] = high_sym_X[2*i, :]
    
    # Create custom low-symmetry molecule
    print("Creating custom low-symmetry molecule...")
    low_sym_X = np.random.normal(0.25, 0.02, (n_atoms, n_features))
    
    # Process both molecules
    result_high = car_system.process_molecule(high_sym_X, true_label=1)
    result_low = car_system.process_molecule(low_sym_X, true_label=0)
    
    print(f"\nHigh-Symmetry Molecule:")
    print(f"  Prediction: {result_high['prediction']}")
    print(f"  Symmetry Score: {result_high['symmetry_score']:.6f}")
    print(f"  Confidence: {result_high['confidence']:.3f}")
    
    print(f"\nLow-Symmetry Molecule:")
    print(f"  Prediction: {result_low['prediction']}")
    print(f"  Symmetry Score: {result_low['symmetry_score']:.6f}")
    print(f"  Confidence: {result_low['confidence']:.3f}")


def example_5_system_statistics():
    """Example 5: Monitor system statistics during processing."""
    print("\n" + "="*60)
    print("Example 5: System Statistics Monitoring")
    print("="*60)
    
    # Initialize CAR system
    car_system = EnhancedCARSystem(n_units=25, n_chunks=3, random_seed=999)
    
    # Generate dataset
    qm9 = QM9Dataset()
    stats = qm9.get_qm9_statistics()
    symmetry_gen = MolecularSymmetryGenerator(stats)
    features, labels = symmetry_gen.generate_complete_dataset(
        n_total=100,
        high_symmetry_ratio=0.25,
        n_atoms=9,
        n_features=5,
        random_seed=999
    )
    
    # Process molecules and track statistics
    print(f"\nProcessing molecules and tracking statistics...")
    for i, (X, y) in enumerate(zip(features, labels)):
        car_system.process_molecule(X, y)
        
        if (i + 1) % 25 == 0:
            stats = car_system.get_statistics()
            print(f"\nAfter {i + 1} molecules:")
            print(f"  Total inferences: {stats['total_inferences']}")
            print(f"  KB hit rate: {stats['kb_hit_rate']:.2%}")
            print(f"  KB size: {stats['knowledge_base_size']}")
            print(f"  KB accuracy: {stats['knowledge_base_accuracy']:.2%}")
            print(f"  Recent accuracy: {stats['recent_accuracy']:.2%}")
            print(f"  Verification rate: {stats['verification_rate']:.2%}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CAR System - Example Usage Demonstrations")
    print("="*60)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_single_experiment()
        example_3_multi_experiment()
        example_4_custom_molecules()
        example_5_system_statistics()
        
        print("\n" + "="*60)
        print("All examples completed successfully! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
