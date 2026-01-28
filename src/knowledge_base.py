#!/usr/bin/env python3
"""
Knowledge Base Module for CAR
Implements pattern storage, retrieval, and management.
Supports Decimal module for extreme precision (Float128 simulation).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
import numpy as np
from decimal import Decimal, getcontext


@dataclass
class KnowledgePattern:
    """
    Knowledge Pattern Data Structure
    
    Attributes:
        features: Feature vector representing the pattern
        prediction: Predicted output value
        ground_truth: Actual target value (for validation)
        confidence: Confidence score (0-1)
        created_at: Creation timestamp
        perspective: Analysis perspective (global/local/uniform)
        is_special: Whether this is a special/diverse pattern
        usage_count: Number of times this pattern has been used
    """
    
    features: np.ndarray
    prediction: float
    ground_truth: float
    confidence: float
    created_at: int
    perspective: str
    is_special: bool = False
    usage_count: int = 1
    
    def __post_init__(self):
        """Validate pattern data after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.usage_count < 1:
            raise ValueError(f"Usage count must be positive, got {self.usage_count}")


class KnowledgeBase:
    """
    Complete Knowledge Base Implementation
    
    Key Features:
    - Cosine similarity-based scoring
    - Multi-scale retrieval thresholds
    - Pattern merging for similar entries
    - Special pattern detection and handling
    - Usage-based weighting
    
    Attributes:
        config: CAR configuration object
        patterns: List of stored knowledge patterns
        created_counter: Counter for pattern IDs
    """
    
    def __init__(self, config):
        """
        Initialize knowledge base.
        
        Args:
            config: CARConfig object with knowledge base parameters
        """
        self.config = config
        self.patterns: List[KnowledgePattern] = []
        self.created_counter = 0
    
    def calculate_unit_score(
        self, 
        x: np.ndarray, 
        pattern: KnowledgePattern,
        use_decimal: bool = False
    ) -> float:
        """
        Calculate unit score for a pattern.
        
        Score = Cosine Similarity × Confidence × Log(Usage) × Time Factor × Diversity
        
        Args:
            x: Current input feature vector
            pattern: Knowledge pattern to score
            use_decimal: Whether to use Decimal for extreme precision (default: False)
            
        Returns:
            Unit score (0-1 range)
        """
        # Empty knowledge base check
        if not self.patterns:
            return 0.5
        
        if use_decimal:
            return self._calculate_unit_score_decimal(x, pattern)
        else:
            return self._calculate_unit_score_float64(x, pattern)
    
    def _calculate_unit_score_float64(
        self, 
        x: np.ndarray, 
        pattern: KnowledgePattern
    ) -> float:
        """Calculate unit score using Float64 (standard precision)."""
        # Cosine similarity computation
        x_norm = np.linalg.norm(x) + 1e-10
        p_norm = np.linalg.norm(pattern.features) + 1e-10
        
        x_normalized = x / x_norm
        p_normalized = pattern.features / p_norm
        
        cosine_sim = np.dot(x_normalized, p_normalized)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # Confidence factor
        confidence_factor = pattern.confidence
        
        # Final score
        if cosine_sim > 0:
            unit_score = cosine_sim * confidence_factor
        else:
            # Negative similarity gets low score but not excluded
            unit_score = 0.1 * confidence_factor
        
        return max(0.0, unit_score)
    
    def _calculate_unit_score_decimal(
        self, 
        x: np.ndarray, 
        pattern: KnowledgePattern
    ) -> float:
        """Calculate unit score using Decimal for extreme precision."""
        try:
            # Convert to Decimal
            x_decimal = np.array([[Decimal(str(val)) for val in row] for row in x.reshape(-1, 1)])
            p_decimal = np.array([[Decimal(str(val)) for val in row] for row in pattern.features.reshape(-1, 1)])
            
            # Calculate norms using Decimal
            x_norm_sq = sum(val * val for val in x_decimal.flatten())
            p_norm_sq = sum(val * val for val in p_decimal.flatten())
            
            x_norm_decimal = x_norm_sq.sqrt() if x_norm_sq > 0 else Decimal('0')
            p_norm_decimal = p_norm_sq.sqrt() if p_norm_sq > 0 else Decimal('0')
            
            # Avoid division by zero
            x_norm_decimal = x_norm_decimal + Decimal('1e-50')
            p_norm_decimal = p_norm_decimal + Decimal('1e-50')
            
            # Normalize
            x_normalized_decimal = x_decimal.flatten() / x_norm_decimal
            p_normalized_decimal = p_decimal.flatten() / p_norm_decimal
            
            # Calculate dot product
            dot_product = sum(x * p for x, p in zip(x_normalized_decimal, p_normalized_decimal))
            
            # Clip to [-1, 1]
            dot_product = max(Decimal('-1'), min(Decimal('1'), dot_product))
            
            # Convert to float for consistency
            cosine_sim = float(dot_product)
            
            # Confidence factor
            confidence_factor = pattern.confidence
            
            # Final score
            if cosine_sim > 0:
                unit_score = cosine_sim * confidence_factor
            else:
                unit_score = 0.1 * confidence_factor
            
            return max(0.0, unit_score)
            
        except Exception as e:
            # Fallback to Float64 if Decimal calculation fails
            return self._calculate_unit_score_float64(x, pattern)
    
    def is_special_pattern(self, features: np.ndarray) -> bool:
        """
        Determine if a pattern is special (dissimilar to existing patterns).
        
        Args:
            features: Feature vector to check
            
        Returns:
            True if pattern is special, False otherwise
        """
        if not self.patterns:
            return True
        
        # Calculate similarity scores to all existing patterns
        all_scores = [
            self.calculate_unit_score(features, p) 
            for p in self.patterns
        ]
        max_score = max(all_scores)
        
        # Check against threshold
        return max_score < self.config.KB_SPECIAL_THRESHOLD
    
    def find_similar_patterns(
        self, 
        features: np.ndarray, 
        threshold: float,
        use_decimal: bool = False
    ) -> List[Tuple[KnowledgePattern, float]]:
        """
        Find patterns similar to input features.
        
        Args:
            features: Input feature vector
            threshold: Minimum similarity threshold (0-1)
            use_decimal: Whether to use Decimal for extreme precision (default: False)
            
        Returns:
            List of (pattern, score) tuples, sorted by score descending
        """
        # Calculate scores for all patterns
        all_patterns = [
            (pattern, self.calculate_unit_score(features, pattern, use_decimal=use_decimal))
            for pattern in self.patterns
        ]
        
        # Sort by score descending
        sorted_patterns = sorted(all_patterns, key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        similar = [(p, s) for p, s in sorted_patterns if s >= threshold]
        
        # If insufficient matches, return top patterns anyway
        if len(similar) < 3:
            similar = sorted_patterns[:max(5, len(sorted_patterns) // 3 + 1)]
        
        return similar
    
    def multi_scale_retrieval(
        self, 
        features: np.ndarray,
        use_decimal: bool = False
    ) -> List[Tuple[KnowledgePattern, float]]:
        """
        Multi-scale retrieval at different threshold levels.
        
        Args:
            features: Input feature vector
            use_decimal: Whether to use Decimal for extreme precision (default: False)
            
        Returns:
            List of (pattern, score) tuples
        """
        # Try each threshold in ascending order
        for threshold in self.config.RETRIEVAL_THRESHOLDS:
            similar = self.find_similar_patterns(features, threshold, use_decimal=use_decimal)
            if len(similar) >= 3:
                return similar
        
        # Fallback: return top patterns by score
        all_patterns = [
            (pattern, self.calculate_unit_score(features, pattern, use_decimal=use_decimal))
            for pattern in self.patterns
        ]
        sorted_patterns = sorted(all_patterns, key=lambda x: x[1], reverse=True)
        
        k = max(5, len(sorted_patterns) // 3)
        return sorted_patterns[:k]
    
    def get_weighted_prediction(
        self, 
        features: np.ndarray,
        use_decimal: bool = False
    ) -> Tuple[float, float]:
        """
        Get weighted prediction from retrieved patterns.
        
        Args:
            features: Input feature vector
            use_decimal: Whether to use Decimal for extreme precision (default: False)
            
        Returns:
            Tuple of (prediction, average_weight)
        """
        retrieved = self.multi_scale_retrieval(features, use_decimal=use_decimal)
        
        if not retrieved:
            return None, 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for pattern, score in retrieved:
            # Multi-factor weight calculation
            weight = score * pattern.confidence * np.log(pattern.usage_count + 1)
            
            # Diversity bonus for special patterns
            if pattern.is_special:
                weight *= (1 + self.config.DIVERSITY_BONUS)
            
            # Temporal decay factor (newer patterns weighted higher)
            time_factor = 1.0 / (1.0 + (self.created_counter - pattern.created_at) * 0.001)
            weight *= time_factor
            
            weighted_sum += weight * pattern.prediction
            total_weight += weight
        
        if total_weight > 0:
            prediction = weighted_sum / total_weight
            avg_weight = total_weight / len(retrieved)
            return prediction, avg_weight
        
        return None, 0.0
    
    def get_statistics(self) -> Dict:
        """
        Get knowledge base statistics.
        
        Returns:
            Dictionary with size, special count, average confidence, total usage
        """
        if not self.patterns:
            return {
                "size": 0,
                "special_count": 0,
                "avg_confidence": 0.0,
                "total_usage": 0
            }
        
        confidences = [p.confidence for p in self.patterns]
        special_count = sum(1 for p in self.patterns if p.is_special)
        
        return {
            "size": len(self.patterns),
            "special_count": special_count,
            "avg_confidence": np.mean(confidences),
            "total_usage": sum(p.usage_count for p in self.patterns)
        }
