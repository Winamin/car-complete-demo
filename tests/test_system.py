"""
Quick test script to verify CAR system functionality.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced_car import EnhancedCARSystem
from src.qm9_dataset import QM9Dataset, MolecularSymmetryGenerator

def test_basic_functionality():
    """Test basic CAR system functionality."""
    print("\n" + "="*60)
    print("Testing CAR System Functionality")
    print("="*60)
    
    # Test 1: QM9 Statistics
    print("\n[Test 1] Loading QM9 statistics...")
    qm9 = QM9Dataset()
    stats = qm9.get_qm9_statistics()
    print(f"  ✓ QM9 statistics loaded successfully")
    print(f"    Mean HOMO-LUMO gap: {stats['mean']:.4f} eV")
    print(f"    Std: {stats['std']:.4f} eV")
    
    # Test 2: Feature Generation
    print("\n[Test 2] Generating molecular features...")
    symmetry_gen = MolecularSymmetryGenerator(stats)
    features, labels = symmetry_gen.generate_complete_dataset(
        n_total=100,  # Small dataset for quick test
        high_symmetry_ratio=0.25,
        n_atoms=9,
        n_features=5,
        random_seed=42
    )
    print(f"  ✓ Generated {len(features)} molecules")
    print(f"    Feature shape: {features.shape}")
    print(f"    Label distribution: {np.bincount(labels)}")
    
    # Test 3: CAR System Initialization
    print("\n[Test 3] Initializing CAR system...")
    car_system = EnhancedCARSystem(n_units=20, n_chunks=2, random_seed=42)
    print(f"  ✓ CAR system initialized")
    print(f"    Units: {car_system.base_system.n_units}")
    print(f"    Chunks: {car_system.base_system.n_chunks}")
    
    # Test 4: Process Single Molecule
    print("\n[Test 4] Processing a single molecule...")
    result = car_system.process_molecule(features[0], labels[0])
    print(f"  ✓ Molecule processed successfully")
    print(f"    Prediction: {result['prediction']}")
    print(f"    True label: {labels[0]}")
    print(f"    Symmetry score: {result['symmetry_score']:.6f}")
    print(f"    Confidence: {result['confidence']:.3f}")
    print(f"    Strategy: {result['strategy']}")
    
    # Test 5: Process Multiple Molecules
    print("\n[Test 5] Processing multiple molecules...")
    correct = 0
    for i in range(min(20, len(features))):
        result = car_system.process_molecule(features[i], labels[i])
        if result['prediction'] == labels[i]:
            correct += 1
    accuracy = correct / min(20, len(features))
    print(f"  ✓ Processed 20 molecules")
    print(f"    Accuracy: {accuracy:.2%}")
    
    # Test 6: System Statistics
    print("\n[Test 6] Getting system statistics...")
    stats = car_system.get_statistics()
    print(f"  ✓ Statistics retrieved successfully")
    print(f"    Total inferences: {stats['total_inferences']}")
    print(f"    Knowledge base size: {stats['knowledge_base_size']}")
    print(f"    KB hit rate: {stats['kb_hit_rate']:.2%}")
    print(f"    Recent accuracy: {stats['recent_accuracy']:.2%}")
    
    print("\n" + "="*60)
    print("All tests passed successfully! ✓")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()