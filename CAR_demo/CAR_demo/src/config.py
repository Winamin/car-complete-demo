#!/usr/bin/env python3
"""
CAR Configuration Dataclasses
Defines all configurable hyperparameters for the CAR architecture.

Date: January 2026
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CARConfig:
    """
    CAR Architecture Hyperparameter Configuration
    
    This dataclass encapsulates all tunable parameters for the CAR system.
    """
    
    # ========================
    # Knowledge Base Settings
    # ========================
    KB_CAPACITY: int = 2000
    """Maximum number of patterns stored in the knowledge base."""
    
    KB_MERGE_THRESHOLD: float = 0.25
    """Similarity threshold for merging patterns (0-1)."""
    
    KB_SPECIAL_THRESHOLD: float = 0.20
    """Similarity threshold for detecting special patterns (0-1)."""
    
    # ========================
    # Learning Parameters
    # ========================
    SUCCESS_THRESHOLD: float = 1.0
    """Threshold for defining successful predictions."""
    
    DIVERSITY_BONUS: float = 0.20
    """Bonus factor for diversity in pattern retrieval (0-1)."""
    
    REVIVAL_BONUS: float = 0.15
    """Bonus for reviving dormant patterns (0-1)."""
    
    CONSENSUS_LEARNING_RATE: float = 0.25
    """Learning rate for consensus updates (0-1)."""
    
    VERIFICATION_LEARNING_RATE: float = 0.1
    """Learning rate for verification score updates (0-1)."""
    
    # ========================
    # Activation & State Parameters
    # ========================
    ACTIVATION_THRESHOLD: float = 0.6
    """Threshold for unit activation (0-1)."""
    
    INITIAL_ACTIVATION: float = 0.1
    """Initial activation value for new units (0-1)."""
    
    INITIAL_VALIDATION: float = 0.5
    """Initial validation score for new patterns (0-1)."""
    
    STATE_CLIP_MIN: float = 0.1
    """Minimum clipping value for state variables (0-1)."""
    
    STATE_CLIP_MAX: float = 0.9
    """Maximum clipping value for state variables (0-1)."""
    
    # ========================
    # Information Processing
    # ========================
    INFO_ACQUIRE_RATE: float = 0.12
    """Rate for acquiring new information (0-1)."""
    
    INFO_CONSOLIDATE_RATE: float = 0.04
    """Rate for consolidating existing information (0-1)."""
    
    LOAD_DECAY_RATE: float = 0.95
    """Decay rate for load balancing (0-1)."""
    
    LOAD_DECISION_THRESHOLD: float = 0.8
    """Threshold for load-based decisions (0-1)."""
    
    # ========================
    # Consensus Parameters
    # ========================
    CONSENSUS_CONFIDENCE_THRESHOLD: float = 0.6
    """Threshold for accepting consensus (0-1)."""
    
    ADJUSTMENT_RATE: float = 0.2
    """Rate for adjustment updates (0-1)."""
    
    # ========================
    # Retrieval Parameters
    # ========================
    RETRIEVAL_THRESHOLDS: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    """Multi-scale retrieval thresholds, sorted ascending."""
    
    UNIFORM_SCALE: float = 1.2
    """Scale factor for uniform perspective weighting."""
    
    LOCAL_TOP_K: float = 0.3
    """Fraction of features for local perspective (0-1)."""
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if all parameters are valid, raises ValueError otherwise.
        """
        # Knowledge base parameters
        if self.KB_CAPACITY <= 0:
            raise ValueError("KB_CAPACITY must be positive")
        
        if not 0 <= self.KB_MERGE_THRESHOLD <= 1:
            raise ValueError("KB_MERGE_THRESHOLD must be in [0, 1]")
        
        if not 0 <= self.KB_SPECIAL_THRESHOLD <= 1:
            raise ValueError("KB_SPECIAL_THRESHOLD must be in [0, 1]")
        
        # Learning parameters
        if not 0 <= self.SUCCESS_THRESHOLD <= 1:
            raise ValueError("SUCCESS_THRESHOLD must be in [0, 1]")
        
        if not 0 <= self.DIVERSITY_BONUS <= 1:
            raise ValueError("DIVERSITY_BONUS must be in [0, 1]")
        
        if not 0 <= self.CONSENSUS_LEARNING_RATE <= 1:
            raise ValueError("CONSENSUS_LEARNING_RATE must be in [0, 1]")
        
        if not 0 <= self.VERIFICATION_LEARNING_RATE <= 1:
            raise ValueError("VERIFICATION_LEARNING_RATE must be in [0, 1]")
        
        # Activation parameters
        if not 0 <= self.ACTIVATION_THRESHOLD <= 1:
            raise ValueError("ACTIVATION_THRESHOLD must be in [0, 1]")
        
        if not 0 <= self.INITIAL_ACTIVATION <= 1:
            raise ValueError("INITIAL_ACTIVATION must be in [0, 1]")
        
        # Consensus parameters
        if not 0 <= self.CONSENSUS_CONFIDENCE_THRESHOLD <= 1:
            raise ValueError("CONSENSUS_CONFIDENCE_THRESHOLD must be in [0, 1]")
        
        # Retrieval thresholds
        for threshold in self.RETRIEVAL_THRESHOLDS:
            if not 0 <= threshold <= 1:
                raise ValueError(f"RETRIEVAL_THRESHOLD {threshold} must be in [0, 1]")
        
        if len(self.RETRIEVAL_THRESHOLDS) != len(set(self.RETRIEVAL_THRESHOLDS)):
            raise ValueError("RETRIEVAL_THRESHOLDS must be unique values")
        
        return True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def get_operational_summary(self) -> dict:
        """
        Get a summary of operational characteristics.
        
        Returns:
            Dictionary with key operational parameters.
        """
        return {
            "knowledge_base_capacity": self.KB_CAPACITY,
            "merge_threshold": self.KB_MERGE_THRESHOLD,
            "special_threshold": self.KB_SPECIAL_THRESHOLD,
            "retrieval_thresholds": self.RETRIEVAL_THRESHOLDS,
            "consensus_threshold": self.CONSENSUS_CONFIDENCE_THRESHOLD,
            "diversity_bonus": self.DIVERSITY_BONUS,
        }
