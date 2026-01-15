# Quick Start Guide

Get up and running with the CAR System in 5 minutes!

## Installation

```bash
# Run the setup script (Windows)
setup.bat
```

Or manually:

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Run Tests

Verify the installation:

```bash
python test_system.py
```

Expected output:
```
============================================================
Testing CAR System Functionality
============================================================

[Test 1] Loading QM9 statistics...
  ✓ QM9 statistics loaded successfully

[Test 2] Generating molecular features...
  ✓ Generated 100 molecules

[Test 3] Initializing CAR system...
  ✓ CAR system initialized

[Test 4] Processing a single molecule...
  ✓ Molecule processed successfully

[Test 5] Processing multiple molecules...
  ✓ Processed 20 molecules

[Test 6] Getting system statistics...
  ✓ Statistics retrieved successfully

============================================================
All tests passed successfully! ✓
============================================================
```

## Run Complete Experiment

Run the full experimental pipeline (10 experiments, 2000 molecules each):

```bash
python experiment.py
```

This will:
- Generate molecular features based on QM9 statistics
- Run 10 independent experiments
- Compute statistical analysis
- Save results to `results/` directory

## Quick Code Example

```python
from enhanced_car import EnhancedCARSystem
from qm9_dataset import QM9Dataset, MolecularSymmetryGenerator
import numpy as np

# Get QM9 statistics
qm9 = QM9Dataset()
stats = qm9.get_qm9_statistics()

# Generate features
symmetry_gen = MolecularSymmetryGenerator(stats)
features, labels = symmetry_gen.generate_complete_dataset(
    n_total=100,
    high_symmetry_ratio=0.25,
    n_atoms=9,
    n_features=5
)

# Initialize and use CAR system
car_system = EnhancedCARSystem(n_units=50, n_chunks=5)

# Process a molecule
result = car_system.process_molecule(features[0], labels[0])
print(f"Prediction: {result['prediction']}")
print(f"Symmetry Score: {result['symmetry_score']:.6f}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Expected Results

Based on the paper, you should see:

- **Accuracy**: ~78% ± 0.6%
- **Precision**: ~75%
- **F1-Score**: ~29% ± 5%
- **Z-Score**: ~-9.2 ± 1.0
- **Cohen's d**: ~-0.54 ± 0.06

## Troubleshooting

### Import Errors

Make sure the virtual environment is activated:
```bash
.venv\Scripts\activate.bat
```

### Dataset Download Issues

If the QM9 dataset download fails (403 error), the system automatically uses pre-defined statistics from the paper. This is normal and doesn't affect functionality.

### Memory Issues

For large experiments, reduce the number of molecules:
```python
# In experiment.py or your script
n_molecules = 500  # Instead of 2000
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the source code in `car_system.py`, `enhanced_car.py`, etc.
- Modify parameters in `experiment.py` to run custom experiments
- Check results in the `results/` directory

## Support

For issues or questions:
1. Check the [README.md](README.md) for detailed documentation
2. Review the test script `test_system.py` for usage examples
3. Examine the paper PDF for theoretical background