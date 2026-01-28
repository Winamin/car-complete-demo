#!/usr/bin/env python3
"""
Complete CAR Model Implementation
Main integration module for the Cognitive Architecture with Retrieval-Based Learning.
Supports Decimal module for extreme precision (Float128 simulation).
"""

import numpy as np
from typing import Dict, List, Tuple
from decimal import Decimal, getcontext
from .config import CARConfig
from .knowledge_base import KnowledgeBase, KnowledgePattern
from .unit import ComputationalUnit, MultiViewAnalyzer


class CompleteCARModel:
    """
    Complete CAR (Cognitive Architecture with Retrieval-Based Learning) Model
    """
    
    def __init__(
        self, 
        config: CARConfig, 
        n_features: int,
        n_units: int = 50
    ):
        """
        Initialize CompleteCARModel.
        
        Args:
            config: CARConfig object with all parameters
            n_features: Dimension of input feature vectors
            n_units: Number of computational units (default 50)
        """
        self.config = config
        self.n_features = n_features
        self.n_units = n_units
        
        # Initialize components
        self.knowledge_base = KnowledgeBase(config)
        self.analyzer = MultiViewAnalyzer(config, n_features)
        
        # Initialize computational units
        self.units = [
            ComputationalUnit(i, config)
            for i in range(n_units)
        ]
        
        # Statistics
        self.special_patterns_detected = 0
        self.consensus_count = 0
        self.prediction_history: List[float] = []
        self.unit_A_history: List[List[float]] = []  # Record A state history
        self.unit_v_history: List[List[float]] = []  # Record v state history
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> 'CompleteCARModel':
        """
        Train CAR by filling knowledge base with patterns.
        """
        for xi, yi in zip(X, y):
            self.knowledge_base.created_counter += 1
            
            is_special = self.knowledge_base.is_special_pattern(xi)
            confidence = 0.9
            
            new_pattern = KnowledgePattern(
                features=xi.copy(),
                prediction=yi,
                ground_truth=yi,
                confidence=confidence,
                created_at=self.knowledge_base.created_counter,
                perspective='fitted',
                is_special=is_special
            )
            
            merged = False
            if not is_special:
                for i, (pattern, score) in enumerate(
                    self.knowledge_base.find_similar_patterns(
                        xi, self.config.KB_MERGE_THRESHOLD
                    )
                ):
                    if score >= self.config.KB_MERGE_THRESHOLD:
                        merge_weight = score
                        new_features = merge_weight * pattern.features + (1 - merge_weight) * xi
                        new_confidence = max(pattern.confidence, confidence)
                        
                        merged_pattern = KnowledgePattern(
                            features=new_features,
                            prediction=merge_weight * pattern.prediction + (1 - merge_weight) * yi,
                            ground_truth=pattern.ground_truth,
                            confidence=new_confidence,
                            created_at=pattern.created_at,
                            perspective='merged',
                            is_special=False,
                            usage_count=pattern.usage_count + 1
                        )
                        self.knowledge_base.patterns[i] = merged_pattern
                        merged = True
                        break
            
            if not merged:
                if len(self.knowledge_base.patterns) >= self.config.KB_CAPACITY:
                    min_score = float('inf')
                    min_idx = 0
                    for i, p in enumerate(self.knowledge_base.patterns):
                        score = p.confidence * np.log(p.usage_count + 1)
                        if score < min_score:
                            min_score = score
                            min_idx = i
                    self.knowledge_base.patterns.pop(min_idx)
                
                self.knowledge_base.patterns.append(new_pattern)
        
        return self
    
    def distributed_discussion(
        self, 
        features: np.ndarray, 
        retrieved: List[Tuple[KnowledgePattern, float]]
    ) -> Tuple[float, float, bool]:
        """
        Distributed discussion for consensus formation.
        """
        if not retrieved:
            return None, 0.0, False
        
        predictions = []
        weights = []
        
        for pattern, score in retrieved:
            weight = score * pattern.confidence * np.log(pattern.usage_count + 1)
            
            if pattern.is_special:
                weight *= (1 + self.config.DIVERSITY_BONUS)
            
            weights.append(weight)
            predictions.append(pattern.prediction)
        
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-10)
        
        consensus_prediction = np.sum(weights * np.array(predictions))
        
        variance = np.sum(
            weights * (np.array(predictions) - consensus_prediction) ** 2
        )
        consensus_confidence = 1.0 / (
            1.0 + np.sqrt(variance) / self.config.SUCCESS_THRESHOLD
        )
        
        consensus_reached = (
            consensus_confidence >= self.config.CONSENSUS_CONFIDENCE_THRESHOLD
        )
        
        if consensus_reached:
            self.consensus_count += 1
        
        return consensus_prediction, consensus_confidence, consensus_reached
    
    def predict(
        self, 
        features: np.ndarray, 
        record_states: bool = True,
        use_decimal: bool = False
    ) -> float:
        """
        Make a prediction for the given input.
        
        Args:
            features: Input feature vector
            record_states: Whether to record unit states for visualization
            use_decimal: Whether to use Decimal for extreme precision (default: False)
            
        Returns:
            Predicted value
        """
        # Retrieve similar patterns (with Decimal support if requested)
        retrieved = self.knowledge_base.multi_scale_retrieval(
            features, 
            use_decimal=use_decimal
        )
        
        # Check for special pattern
        is_special = self.knowledge_base.is_special_pattern(features)
        if is_special:
            self.special_patterns_detected += 1
        
        # Try distributed discussion
        consensus_pred, confidence, consensus_reached = (
            self.distributed_discussion(features, retrieved)
        )
        
        if consensus_reached and consensus_pred is not None:
            self.prediction_history.append(consensus_pred)
        else:
            kb_pred, kb_confidence = self.knowledge_base.get_weighted_prediction(
                features,
                use_decimal=use_decimal
            )
            if kb_pred is not None:
                consensus_pred = kb_pred
                confidence = kb_confidence
                self.prediction_history.append(kb_pred)
            else:
                self.prediction_history.append(0.0)
                consensus_pred = 0.0
                confidence = 0.0
        
        # Update unit states (A, v) - based on prediction results
        if record_states:
            self._update_unit_states(consensus_pred, confidence)
        
        return consensus_pred
    
    def _update_unit_states(self, prediction: float, confidence: float):
        """
        Update all computational unit states A and v
        
        A (Activation): Adjusted based on prediction confidence
        v (Validation): Adjusted based on historical performance
        """
        # Success flag - based on confidence
        success = confidence >= self.config.CONSENSUS_CONFIDENCE_THRESHOLD
        
        # Update each unit
        for unit in self.units:
            # Update validation score v
            unit.update_validation(success)
            
            # Adjust activation state A - based on validation score
            target_A = self.config.ACTIVATION_THRESHOLD if success else 0.5
            lr = self.config.CONSENSUS_LEARNING_RATE
            unit.activation = (1 - lr) * unit.activation + lr * target_A
            
            # Clip to valid range
            unit.activation = np.clip(
                unit.activation,
                self.config.STATE_CLIP_MIN,
                self.config.STATE_CLIP_MAX
            )
        
        # Record state history
        self.unit_A_history.append([u.activation for u in self.units])
        self.unit_v_history.append([u.validation for u in self.units])
    
    def get_prediction_statistics(self) -> Dict:
        """Get statistics about predictions made."""
        if not self.prediction_history:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "unique_values": 0
            }
        
        values = [round(v, 10) for v in self.prediction_history]
        
        return {
            "count": len(self.prediction_history),
            "mean": np.mean(self.prediction_history),
            "std": np.std(self.prediction_history),
            "unique_values": len(set(values))
        }
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get knowledge base statistics."""
        return self.knowledge_base.get_statistics()
    
    def get_unit_statistics(self) -> Dict:
        """Get computational unit statistics (A, v states)."""
        if not self.units:
            return {"total": 0, "active": 0, "avg_A": 0.0, "avg_v": 0.0}
        
        activations = [u.activation for u in self.units]
        validations = [u.validation for u in self.units]
        
        return {
            "total": len(self.units),
            "active": sum(1 for u in self.units if u.activation > 0.5),
            "avg_A": np.mean(activations),
            "avg_v": np.mean(validations),
            "A_range": [min(activations), max(activations)],
            "v_range": [min(validations), max(validations)],
            "A_history": activations,
            "v_history": validations,
            "A_history_time": self.unit_A_history,
            "v_history_time": self.unit_v_history
        }
    
    def get_model_summary(self) -> Dict:
        """Get a complete summary of the model state."""
        return {
            "configuration": self.config.get_operational_summary(),
            "knowledge_base": self.get_knowledge_base_stats(),
            "predictions": self.get_prediction_statistics(),
            "units": {
                "total": self.n_units,
                "active": sum(1 for u in self.units if u.activation > 0.5),
                "avg_A": np.mean([u.activation for u in self.units]),
                "avg_v": np.mean([u.validation for u in self.units])
            },
            "special_patterns_detected": self.special_patterns_detected,
            "consensus_count": self.consensus_count
        }
    
    def reset(self):
        """Reset the model state for a new experiment."""
        self.knowledge_base = KnowledgeBase(self.config)
        self.prediction_history = []
        self.unit_A_history = []
        self.unit_v_history = []
        self.special_patterns_detected = 0
        self.consensus_count = 0
        for unit in self.units:
            unit.activation = self.config.INITIAL_ACTIVATION
            unit.validation = self.config.INITIAL_VALIDATION
            unit.prediction_history = []
            unit.success_count = 0
            unit.total_predictions = 0
    
    def get_all_states(self) -> Dict:
        """
        Get all unit states (A, v) and feature statistics.
        
        Returns:
            Dictionary with A, v states and X (feature) info
        """
        A_states = [u.activation for u in self.units]
        v_states = [u.validation for u in self.units]
        
        # Get feature statistics from knowledge base
        if self.knowledge_base.patterns:
            all_features = np.array([p.features for p in self.knowledge_base.patterns])
            X_stats = {
                "n_patterns": len(self.knowledge_base.patterns),
                "feature_dim": self.n_features,
                "feature_mean": float(np.mean(all_features)),
                "feature_std": float(np.std(all_features)),
                "feature_min": float(np.min(all_features)),
                "feature_max": float(np.max(all_features))
            }
        else:
            X_stats = {
                "n_patterns": 0,
                "feature_dim": self.n_features,
                "feature_mean": 0.0,
                "feature_std": 0.0,
                "feature_min": 0.0,
                "feature_max": 0.0
            }
        
        return {
            "A": A_states,           # Activation states
            "v": v_states,           # Validation states
            "X": X_stats,            # Feature statistics
            "prediction_history": self.prediction_history,
            "A_history_time": self.unit_A_history,
            "v_history_time": self.unit_v_history
        }
