# CAR System: Knowledge-Driven Gradient-Free Optimization

## Overview

CAR (Compare-Adjust-Record) is a novel computational architecture for property prediction through iterative unit interactions without gradient-based optimization. This system demonstrates that intelligent behavior can emerge from simple local interaction protocols between autonomous computational units.

### Core Philosophy

The CAR system is fundamentally different from traditional neural networks:

- **No Gradient Descent**: Learning emerges from local interactions, not backpropagation
- **Bounded Communication**: Units communicate via tanh-bounded signals in (-1, 1)
- **Simultaneous Learning-Prediction**: No separation between training and testing phases
- **Explicit Knowledge Storage**: Patterns stored in a retrievable knowledge base
- **White-Box Architecture**: Every prediction is fully interpretable

## Architecture

The CAR system consists of five core mechanisms working in concert:

### 1. Compare

Computational units analyze input features and compare them against stored knowledge patterns. Each unit maintains independent feature weights, enabling diverse perspectives on the same input.

### 2. Adjust

Based on comparison results, units adjust their internal states. The adjustment is guided by knowledge base matches, with learning rate modulated by similarity strength and historical success rates.

### 3. Record

Successful prediction patterns are stored in the knowledge base for future use. The system maintains a dynamic balance between creating new patterns and merging similar ones.

### 4. Discuss

Multiple units participate in distributed discussion to reach consensus. Unit contributions are weighted by their historical performance, enabling robust ensemble predictions.

### 5. Reflect

The system periodically reflects on recent performance and adapts its learning strategy. This includes adjusting learning rates based on error trends.

## Key Features

### Multi-Scale Similarity Retrieval

The knowledge base query operates across multiple similarity thresholds to find relevant patterns:

- Coarse filtering identifies broadly similar cases
- Fine filtering refines to highly specific matches
- Medium thresholds provide balanced retrieval

### Weighted Consensus Discussion

Unit contributions to consensus are weighted by:

- Historical success rate
- Current confidence level
- Knowledge base influence

### Adaptive Learning Rate

The system continuously adjusts its learning rate:

- Decreases when performance is good
- Increases when errors exceed threshold
- Maintains optimal adaptation speed

### Error-Based Knowledge Management

The knowledge base implements intelligent forgetting:

- Low-utility patterns are removed when capacity is exceeded
- Utility considers success rate, recency, and average error
- Ensures memory quality over quantity

## System Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_units | 20 | Number of computational units |
| feature_dim | 71 | Dimensionality of input features |
| learning_rate | 0.3 | Initial learning rate |
| consensus_threshold | 0.6 | Minimum confidence for consensus |
| success_threshold | 1.0 eV | Error threshold for success |

### Knowledge Base Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| kb_capacity | 500 | Maximum patterns in knowledge base |
| similarity_thresholds | [0.3, 0.5, 0.7] | Multi-scale retrieval thresholds |
| pattern_merge_threshold | 0.80 | Similarity threshold for merging |
| reflection_interval | 30 | Iterations between reflections |

## Mathematical Foundation

### Knowledge-Driven Gradient Estimation

The system updates parameters using knowledge-driven gradient estimation:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha \cdot \nabla_{KD}\mathcal{L}(\mathbf{x}_t, \mathcal{K})$$

where $\nabla_{KD}$ represents the knowledge-driven gradient computed from similar historical cases.

### Hypothesis Generation

Hypotheses are generated from knowledge base matches with confidence:

$$\mathcal{H} = [\hat{y}_{pred}, v_{conf}, w_{sim}]$$

### Adaptive Learning Rate

$$\alpha_{t+1} = \begin{cases}
\alpha_t \cdot 0.95 & \text{if } \bar{e} < \theta_{success} \\
\min(0.5, \alpha_t / 0.95) & \text{if } \bar{e} \geq \theta_{success}
\end{cases}$$

## Usage

### Basic Usage

```python
from car_system import CARSystem

# Initialize CAR system
car = CARSystem(
    num_units=20,
    feature_dim=71,
    kb_capacity=500,
    learning_rate=0.3,
    consensus_threshold=0.6,
    similarity_thresholds=[0.3, 0.5, 0.7],
    pattern_merge_threshold=0.80,
    reflection_interval=30,
    success_threshold=1.0,
    exploration_value=7.5
)

# Process samples (with internal feedback learning)
for features, target in zip(X, y):
    result = car.infer(features, target)
    print(f"Prediction: {result['prediction']:.3f} eV")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Knowledge Base Size: {result['knowledge_size']}")

# Get system statistics
stats = car.get_statistics()
```

### Running Experiments

```bash
python src/car_system.py
```

This will run the complete experiment pipeline with synthetic data and report performance metrics.

## What CAR Does NOT Use

The CAR system eliminates all traditional AI training machinery:

- No gradient descent
- No loss function
- No backpropagation
- No weight updates through optimization
- No separate training phase
- No explicit target functions for optimization

## Project Structure

```
car-complete-demo/
├── src/
│   ├── __init__.py
│   └── car_system.py              # Complete CAR implementation
├── data/
│   ├── gdb9.sdf                   # QM9 molecular structures
│   └── gdb9.sdf.csv               # QM9 properties
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Yingxu Wang**
- Email: wangwang228879@163.com