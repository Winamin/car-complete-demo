#!/usr/bin/env python3
"""
Computational Unit Module for CAR
Implements autonomous computational units with multi-view analysis.
"""

import numpy as np
from typing import Dict, List
from .config import CARConfig


class ComputationalUnit:
    """
    Autonomous Computational Unit for CAR
    
    Represents a single unit in the CAR architecture with:
    - Activation weight (A): Current activation level
    - Validation score (v): Historical prediction accuracy
    - Feature vector (x): Data sample representation
    
    Attributes:
        unit_id: Unique identifier for this unit
        config: CAR configuration object
        activation: Current activation level (0-1)
        validation: Validation score (0-1)
        features: Stored feature vector
        prediction_history: History of prediction outcomes
        success_count: Number of successful predictions
        total_predictions: Total predictions made
    """
    
    def __init__(
        self, 
        unit_id: int,
        config: CARConfig,
        initial_activation: float = None,
        initial_validation: float = None
    ):
        """
        Initialize a computational unit.
        
        Args:
            unit_id: Unique identifier for this unit
            config: CAR configuration object
            initial_activation: Starting activation (uses config default if None)
            initial_validation: Starting validation (uses config default if None)
        """
        self.unit_id = unit_id
        self.config = config
        
        # State variables (A, v, x)
        self.activation = initial_activation or config.INITIAL_ACTIVATION
        self.validation = initial_validation or config.INITIAL_VALIDATION
        self.features = None
        
        # Interaction history for validation tracking
        self.prediction_history: List[bool] = []
        self.success_count = 0
        self.total_predictions = 0
    
    @property
    def tanh_activation(self) -> float:
        """Get bounded activation through tanh transformation."""
        return np.tanh(self.activation)
    
    def update_validation(self, success: bool):
        """
        Update validation score based on prediction success.
        
        Uses exponential moving average for smooth updates.
        
        Args:
            success: Whether the prediction was successful
        """
        self.prediction_history.append(success)
        if success:
            self.success_count += 1
        self.total_predictions += 1
        
        # Update validation score
        learning_rate = self.config.VERIFICATION_LEARNING_RATE
        target = 1.0 if success else 0.0
        
        self.validation = (1 - learning_rate) * self.validation + learning_rate * target
    
    def get_validation_resilience(self) -> float:
        """
        Calculate validation resilience score.
        
        Returns:
            Validation resilience score in [0, 1]
        """
        if self.total_predictions == 0:
            return self.validation
        
        historical_rate = self.success_count / self.total_predictions
        
        # Combine current validation with history
        resilience = (
            self.config.CONSENSUS_LEARNING_RATE * self.validation +
            (1 - self.config.CONSENSUS_LEARNING_RATE) * historical_rate
        )
        
        return np.clip(resilience, 0, 1)
    
    def compare_with(self, other: 'ComputationalUnit') -> float:
        """
        Compare this unit with another unit.
        
        Computes absolute difference in tanh activations.
        
        Args:
            other: Another computational unit
            
        Returns:
            Difference measure (0 = identical, 2 = maximally different)
        """
        return abs(self.tanh_activation - other.tanh_activation)
    
    def adjust_activation(
        self, 
        peer_activations: List[float], 
        learning_rate: float = None
    ):
        """
        Adjust activation based on peer influence.
        
        Args:
            peer_activations: List of peer tanh activation values
            learning_rate: Learning rate (uses config default if None)
        """
        if not peer_activations:
            return
        
        lr = learning_rate or self.config.CONSENSUS_LEARNING_RATE
        
        # Target is average of peer activations
        peer_avg = np.mean(peer_activations)
        
        # Update towards peer average
        self.activation = (1 - lr) * self.activation + lr * peer_avg
        
        # Clip to valid range
        self.activation = np.clip(
            self.activation, 
            self.config.STATE_CLIP_MIN, 
            self.config.STATE_CLIP_MAX
        )
    
    def get_state(self) -> Dict:
        """Get current unit state (A, v, x)."""
        return {
            'unit_id': self.unit_id,
            'A': self.activation,           # Activation state
            'v': self.validation,           # Validation state
            'tanh_A': self.tanh_activation, # Bounded activation
            'success_rate': self.success_count / max(1, self.total_predictions),
            'total_predictions': self.total_predictions,
            'has_features': self.features is not None
        }
    
    def reset(self):
        """Reset unit to initial state."""
        self.activation = self.config.INITIAL_ACTIVATION
        self.validation = self.config.INITIAL_VALIDATION
        self.features = None
        self.prediction_history = []
        self.success_count = 0
        self.total_predictions = 0
    
    def __repr__(self) -> str:
        """String representation of the unit."""
        return (
            f"Unit(id={self.unit_id}, A={self.activation:.3f}, "
            f"v={self.validation:.3f}, predictions={self.total_predictions})"
        )


class MultiViewAnalyzer:
    """
    Multi-View Feature Analysis for CAR
    
    Analyzes features from multiple perspectives:
    - Global: All features with equal weight
    - Local: Top-k most important features
    - Uniform: All features uniformly weighted
    """
    
    def __init__(self, config: CARConfig, n_features: int):
        """
        Initialize multi-view analyzer.
        
        Args:
            config: CAR configuration object
            n_features: Number of input features
        """
        self.config = config
        self.n_features = n_features
        
        # Initialize perspective weights
        self.global_weight = np.ones(n_features) / np.sqrt(n_features)
        self.local_weight = np.zeros(n_features)
        self.uniform_weight = np.ones(n_features) * config.UNIFORM_SCALE
        
        # Set local weights for top-k features
        top_k = int(n_features * config.LOCAL_TOP_K)
        self.local_weight[:top_k] = 1.0 / np.sqrt(top_k)
    
    def perspective_similarity(
        self, 
        x1: np.ndarray, 
        x2: np.ndarray, 
        perspective: str
    ) -> float:
        """
        Calculate similarity from a specific perspective.
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            perspective: 'global', 'local', or 'uniform'
            
        Returns:
            Weighted cosine similarity
        """
        if perspective == 'global':
            w = self.global_weight
        elif perspective == 'local':
            w = self.local_weight
        else:
            w = self.uniform_weight
        
        # Apply perspective weights
        x1_weighted = x1 * w
        x2_weighted = x2 * w
        
        # Cosine similarity
        dot = np.dot(x1_weighted, x2_weighted)
        norm1 = np.linalg.norm(x1_weighted) + 1e-10
        norm2 = np.linalg.norm(x2_weighted) + 1e-10
        
        return dot / (norm1 * norm2)
    
    def multi_perspective_analysis(
        self, 
        x: np.ndarray, 
        patterns: List
    ) -> Dict[str, float]:
        """
        Analyze features from multiple perspectives.
        
        Args:
            x: Input feature vector
            patterns: List of (pattern, score) tuples
            
        Returns:
            Dictionary with similarity scores for each perspective
        """
        if not patterns:
            return {'global': 0.0, 'local': 0.0, 'uniform': 0.0}
        
        perspectives = {
            'global': 0.0,
            'local': 0.0,
            'uniform': 0.0
        }
        
        # Calculate similarities from each perspective
        for pattern, _ in patterns:
            perspectives['global'] += self.perspective_similarity(
                x, pattern.features, 'global'
            )
            perspectives['local'] += self.perspective_similarity(
                x, pattern.features, 'local'
            )
            perspectives['uniform'] += self.perspective_similarity(
                x, pattern.features, 'uniform'
            )
        
        # Average over patterns
        n = len(patterns)
        for key in perspectives:
            perspectives[key] /= n
        
        return perspectives
    
    def get_perspective_weights(self) -> Dict[str, np.ndarray]:
        """Get the weight vectors for each perspective."""
        return {
            'global': self.global_weight,
            'local': self.local_weight,
            'uniform': self.uniform_weight
        }
