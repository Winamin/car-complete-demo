# CAR System: Compare-Adjust-Record Computational Architecture

A complete implementation of the Compare-Adjust-Record (CAR) mechanism for emergent pattern detection through iterative computational unit interactions with bounded state dynamics.

## Overview

This project implements a novel computational architecture fundamentally different from deep neural networks, where autonomous computational units maintain independent state representations and interact through CAR cycles without gradient-based optimization or backpropagation.

### Key Features

- **Non-Gradient Architecture**: No gradient descent, loss functions, or backpropagation
- **Bounded Signal Transmission**: Communication via tanh-bounded signals in (-1, 1)
- **Autonomous Hyperparameter Adaptation**: Self-modifying learning rates
- **Distributed Consensus**: Emerges from local interactions
- **Enhanced Mechanisms**: Knowledge base learning, hypothesis verification, distributed discussion, and reflection

## Based On

This implementation is based on the paper:
> *Emergent Pattern Detection Through Iterative Computational Unit Interactions with Bounded State Dynamics* by Yingxu Wang (January 2026)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Run the setup script (Windows):
```bash
setup.bat
```

Or manually set up:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate.bat
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
NearOi/
├── car_system.py              # Core CAR system implementation
├── enhanced_car.py            # Enhanced mechanisms (KB, Hypothesis, Discussion, Reflection)
├── qm9_dataset.py             # QM9 dataset download and processing
├── experiment.py              # Experimental pipeline with statistical analysis
├── requirements.txt           # Python dependencies
├── setup.bat                  # Windows setup script
├── README.md                  # This file
├── data/                      # Directory for QM9 dataset (auto-created)
└── results/                   # Directory for experiment results (auto-created)
```

## Usage

### Quick Start

Run the complete experimental pipeline:

```bash
python experiment.py
```

This will:
1. Download QM9 dataset statistics (or use pre-defined values)
2. Generate molecular features based on real QM9 properties
3. Run 10 independent experiments
4. Compute statistical analysis (Z-score, Cohen's d, etc.)
5. Save results to `results/` directory

### Basic Usage

```python
from enhanced_car import EnhancedCARSystem
from qm9_dataset import QM9Dataset, MolecularSymmetryGenerator

# Get QM9 statistics
qm9 = QM9Dataset()
qm9_stats = qm9.get_qm9_statistics()

# Generate molecular features
symmetry_gen = MolecularSymmetryGenerator(qm9_stats)
features, labels = symmetry_gen.generate_complete_dataset(
    n_total=2000,
    high_symmetry_ratio=0.25,
    n_atoms=9,
    n_features=5
)

# Initialize CAR system
car_system = EnhancedCARSystem(n_units=50, n_chunks=5)

# Process a molecule
result = car_system.process_molecule(features[0], labels[0])
print(f"Prediction: {result['prediction']}")
print(f"Symmetry Score: {result['symmetry_score']}")
print(f"Confidence: {result['confidence']}")
```

### Advanced Usage

Run custom experiments:

```python
from experiment import ExperimentRunner

runner = ExperimentRunner(n_units=50, n_chunks=5)

# Run single experiment
results = runner.run_single_experiment(features, labels, random_seed=42)

# Run multiple experiments
multi_results = runner.run_multi_experiment(
    n_experiments=10,
    n_molecules=2000,
    high_symmetry_ratio=0.25,
    base_seed=42
)

# Save results
runner.save_results(multi_results, filename="my_experiment.json")
```

## System Components

### 1. Core CAR System (`car_system.py`)

Implements the fundamental Compare-Adjust-Record cycle:
- **Compare**: Compute pairwise similarities between units
- **Adjust**: Update activation weights through consensus
- **Record**: Accumulate interaction history

### 2. Enhanced Mechanisms (`enhanced_car.py`)

- **Knowledge Base**: Stores high-value cases for reuse
- **Hypothesis Verification**: Generates and tests hypotheses
- **Distributed Discussion**: Multi-unit consensus building
- **Reflection**: Self-validation and improvement

### 3. QM9 Dataset Handler (`qm9_dataset.py`)

- Downloads real QM9 molecular dataset (133,885 molecules)
- Extracts authentic HOMO-LUMO gap statistics
- Generates features with controlled symmetry properties

### 4. Experimental Pipeline (`experiment.py`)

- Runs multi-experiment validation
- Computes statistical metrics (Z-score, Cohen's d, discrimination index)
- Evaluates classification performance (accuracy, precision, recall, F1)

## Experimental Results

Based on 10 independent experiments with QM9-based data:

| Metric | Mean ± Std |
|--------|-----------|
| Accuracy | 78.01% ± 0.57% |
| Precision | 75.14% |
| Recall | 18.46% |
| F1-Score | 29.30% ± 5.20% |
| Z-Score | -9.19 ± 1.01 |
| Cohen's d | -0.54 ± 0.06 |
| Discrimination Index | 0.54 ± 0.06 |

### Interpretation

- **High Precision**: When the system identifies a molecule as high-symmetry, it's correct ~75% of the time
- **Conservative Classification**: System only labels molecules with clear symmetry signals
- **Statistical Significance**: Z-score of -9.19 indicates extremely significant differentiation (p ≪ 0.001)
- **Medium Effect Size**: Cohen's d of -0.54 indicates meaningful practical difference

## Configuration Parameters

### Core CAR Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `n_units` | Number of computational units | 50 |
| `n_chunks` | Number of chunks for distributed processing | 5 |
| `alpha` | Tanh input scale | 0.5 |
| `beta` | Consensus learning rate | 0.25 |
| `theta_sat` | Activation threshold | 0.6 |
| `theta_c` | Convergence threshold | 0.06 |
| `tau_min` | Minimum iterations | 30 |
| `tau_max` | Maximum iterations | 400 |

### Enhanced Mechanism Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `KB_capacity` | Knowledge base max entries | 5000 |
| `KB_similarity_threshold` | Knowledge base match threshold | 0.85 |
| `verification_threshold` | Hypothesis acceptance threshold | 0.6 |
| `max_discussion_rounds` | Maximum discussion rounds | 5 |
| `consensus_threshold` | Consensus agreement threshold | 0.7 |
| `reflection_interval` | Inferences between reflections | 50 |

### Dataset Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `n_molecules` | Total molecules per experiment | 2000 |
| `high_symmetry_ratio` | Proportion of high-symmetry molecules | 0.25 |
| `n_atoms` | Atoms per molecule (from QM9) | 9 |
| `n_features` | Orbital features per atom | 5 |

## QM9 Statistics

The system uses authentic statistical properties from the real QM9 dataset:

| Statistic | Value | Unit |
|-----------|-------|------|
| Total Molecules | 133,885 | - |
| Mean HOMO-LUMO Gap | 0.2511 | eV |
| Standard Deviation | 0.0475 | eV |
| Minimum | 0.0246 | eV |
| Maximum | 0.6221 | eV |

## Molecular Symmetry Types

High-symmetry molecules include these point groups:
- **C2v**: Water-like (2-fold rotation + 2 vertical mirrors)
- **C2h**: Antisymmetric mirrors with inversion center
- **Cs**: Single mirror plane
- **Ci**: Inversion center only
- **Cn**: n-fold rotation axis (cyclic symmetry)

## Output Files

Results are saved to the `results/` directory as JSON files with timestamps:

```json
{
  "mean_accuracy": 0.7801,
  "std_accuracy": 0.0057,
  "mean_f1": 0.2930,
  "std_f1": 0.0520,
  "mean_z_score": -9.19,
  "mean_cohens_d": -0.54,
  "total_time": 8.22,
  "all_results": [...]
}
```

## License

MIT License

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{wang2026emergent,
  title={Emergent Pattern Detection Through Iterative Computational Unit Interactions with Bounded State Dynamics},
  author={Wang, Yingxu},
  journal={arXiv preprint},
  year={2026}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact:
- **Author**: Yingxu Wang
- **Email**: yingxuw814@gmail.com

## Acknowledgments

This work builds on the QM9 dataset (Ruddigkeit et al., 2012) and implements the theoretical framework described in the original CAR paper.

---

**Note**: This is a research implementation focused on demonstrating the CAR mechanism's capabilities. The system is designed for scientific validation and may require adaptation for production use cases.