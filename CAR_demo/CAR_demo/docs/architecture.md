# CAR Architecture Documentation

## Overview

CAR (Cognitive Architecture with Retrieval-Based Learning) is a novel machine learning framework that achieves remarkable noise robustness through a fundamentally different computational paradigm.

## Core Concepts

### Autonomous Computational Units

Each CAR unit maintains three state variables:

- **Activation Weight (A)**: Current activation level [0, 1]
- **Validation Score (v)**: Historical prediction accuracy [0, 1]  
- **Feature Vector (x)**: Data sample representation

### Compare-Adjust-Record Cycle

1. **Compare**: Units compute similarity between states
2. **Adjust**: Units update activation based on peer influence
3. **Record**: Units store interaction outcomes for future reference

### Knowledge Base

The knowledge base stores validated patterns for retrieval during inference:

- **Pattern Storage**: Features, predictions, and metadata
- **Multi-Scale Retrieval**: Different threshold levels for pattern matching
- **Pattern Merging**: Similar patterns are merged to avoid redundancy
- **Special Pattern Detection**: Dissimilar patterns get diversity bonuses

## Architecture Diagram

```
                    ┌─────────────────────────┐
                    │   Input Feature Vector  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Multi-Scale Retrieval  │
                    │  (Knowledge Base)       │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
   ┌────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
   │ Distributed     │ │ Special       │ │ Weighted        │
   │ Discussion      │ │ Pattern       │ │ Retrieval       │
   │ (Consensus)     │ │ Detection     │ │ (Fallback)      │
   └────────┬────────┘ └───────┬───────┘ └────────┬────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Final Prediction      │
                    └─────────────────────────┘
```

## Key Differences from Deep Learning

| Aspect | Deep Learning | CAR |
|--------|---------------|-----|
| Optimization | Gradient descent | Local interactions |
| Parameters | Millions | 0 |
| Training | Required | Simultaneous learn-infer |
| Noise Response | Accumulates | Filters through validation |
| Interpretability | Black-box | White-box |

## Noise Robustness Mechanism

CAR's noise robustness comes from **multi-factor weighting**:

```
Weight = Similarity × Confidence × log(Usage + 1) × TimeFactor × DiversityBonus
```

This combination provides robust differentiation even when:
- Cosine similarity becomes random (noise > 10⁷⁵)
- Individual factors provide limited information

The key insight: **when any single factor degrades, others compensate**.

## Operational Regions

| Region | Noise Multiplier | Behavior |
|--------|-----------------|----------|
| Normal | 10⁶ - 10¹⁵⁰ | Genuine pattern recognition |
| Near Limit | 10¹⁵⁰ - 10¹⁹⁰ | Degraded but functional |
| Warning | 10¹⁹⁰ - 10²⁰⁰ | Numerical warnings |
| Collapse | > 10²⁰⁰ | Float64 overflow |

## Configuration Parameters

### Knowledge Base Settings

- `KB_CAPACITY`: Maximum patterns (default: 2000)
- `KB_MERGE_THRESHOLD`: Similarity threshold for merging (default: 0.25)
- `KB_SPECIAL_THRESHOLD`: Threshold for special patterns (default: 0.20)

### Learning Parameters

- `DIVERSITY_BONUS`: Bonus for diverse patterns (default: 0.20)
- `CONSENSUS_LEARNING_RATE`: Consensus update rate (default: 0.25)
- `VERIFICATION_LEARNING_RATE`: Validation update rate (default: 0.10)

### Retrieval Parameters

- `RETRIEVAL_THRESHOLDS`: Multi-scale thresholds (default: [0.1, 0.2, 0.3])
- `UNIFORM_SCALE`: Uniform perspective weight (default: 1.2)
- `LOCAL_TOP_K`: Local perspective fraction (default: 0.3)

## Performance Characteristics

### Computational Complexity

- **Training**: O(n × KB_SIZE) per sample for pattern addition/merging
- **Inference**: O(KB_SIZE) for multi-scale retrieval
- **Memory**: O(KB_SIZE × FEATURE_DIM)

### Scalability

- Feature dimension: Any (practical limit based on computation)
- Knowledge base: Configurable capacity
- Units: Configurable number (typically 50-100)

## Limitations

1. **Feature Dependence**: Performance depends on informative features
2. **Similarity Metric**: Requires appropriate similarity measures
3. **Numerical Limits**: Float64 overflow at ~10²⁰⁰ noise
4. **Cold Start**: Requires some initial interactions
