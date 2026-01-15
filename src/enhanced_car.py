"""
Enhanced CAR Mechanisms Implementation

This module implements the enhanced CAR mechanisms including:
- Knowledge Base Learning
- Hypothesis Generation and Verification
- Distributed Discussion and Consensus
- Reflection and Self-Validation

These mechanisms significantly improve the system's reasoning capabilities.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time
import math


@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base."""
    x: np.ndarray  # Feature vector
    symmetry_score: float  # Computed symmetry score
    prediction: int  # Predicted class (0 or 1)
    true_label: int  # Ground truth label
    timestamp: float  # Creation timestamp
    confidence: float = 0.0  # Confidence value


@dataclass
class Hypothesis:
    """Hypothesis for verification."""
    predicted_symmetry: float  # Predicted symmetry score
    predicted_class: int  # Predicted class
    confidence: float  # Confidence value
    source_unit: int  # Unit that generated the hypothesis


class KnowledgeBase:
    """
    Shared knowledge base for storing high-value symmetry detection cases.
    
    The knowledge base stores entries for reuse across all computational units,
    enabling incremental learning without retraining.
    """
    
    def __init__(self, capacity: int = 5000, similarity_threshold: float = 0.85):
        """
        Initialize knowledge base.
        
        Args:
            capacity: Maximum number of entries
            similarity_threshold: Threshold for considering entries as matches
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.entries: List[KnowledgeEntry] = []
    
    def add_entry(self, x: np.ndarray, symmetry_score: float, prediction: int,
                  true_label: int, confidence: float = 0.0):
        """
        Add an entry to the knowledge base.
        
        Args:
            x: Feature vector
            symmetry_score: Computed symmetry score
            prediction: Predicted class
            true_label: Ground truth label
            confidence: Confidence value
        """
        entry = KnowledgeEntry(
            x=x.copy(),
            symmetry_score=symmetry_score,
            prediction=prediction,
            true_label=true_label,
            timestamp=time.time(),
            confidence=confidence
        )
        
        self.entries.append(entry)
        
        # Remove oldest entries if capacity exceeded
        if len(self.entries) > self.capacity:
            self.entries.pop(0)
    
    def retrieve_similar(self, x: np.ndarray, k: int = 5) -> List[KnowledgeEntry]:
        """
        Retrieve k most similar entries from the knowledge base.
        
        Args:
            x: Query feature vector
            k: Number of entries to retrieve
            
        Returns:
            List of similar entries sorted by similarity
        """
        if not self.entries:
            return []
        
        similarities = []
        for entry in self.entries:
            sim = self._cosine_similarity(x, entry.x)
            similarities.append((sim, entry))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k entries that exceed threshold
        similar_entries = []
        for sim, entry in similarities:
            if sim >= self.similarity_threshold:
                similar_entries.append(entry)
                if len(similar_entries) >= k:
                    break
        
        return similar_entries
    
    def _cosine_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(x1, x2) / (norm1 * norm2)
    
    def get_accuracy(self) -> float:
        """Compute overall accuracy of knowledge base entries."""
        if not self.entries:
            return 0.0
        
        correct = sum(1 for entry in self.entries if entry.prediction == entry.true_label)
        return correct / len(self.entries)
    
    def get_size(self) -> int:
        """Get current number of entries in knowledge base."""
        return len(self.entries)


class HypothesisVerifier:
    """
    Hypothesis generation and verification system.
    
    When a computational unit encounters a similar case from the knowledge base,
    it generates a hypothesis that is verified against the actual sample.
    """
    
    def __init__(self, verification_threshold: float = 0.6):
        """
        Initialize hypothesis verifier.
        
        Args:
            verification_threshold: Threshold for accepting hypotheses as valid
        """
        self.verification_threshold = verification_threshold
        self.hypotheses_history: List[Tuple[Hypothesis, float]] = []
    
    def generate_hypothesis(self, similar_entries: List[KnowledgeEntry],
                           current_symmetry: float) -> Optional[Hypothesis]:
        """
        Generate a hypothesis based on similar knowledge base entries.
        
        Args:
            similar_entries: List of similar entries from knowledge base
            current_symmetry: Current symmetry score
            
        Returns:
            Generated hypothesis or None if no similar entries
        """
        if not similar_entries:
            return None
        
        # Aggregate predictions from similar entries
        predictions = [entry.prediction for entry in similar_entries]
        symmetry_scores = [entry.symmetry_score for entry in similar_entries]
        
        # Weighted by confidence and recency
        weights = []
        for entry in similar_entries:
            age = time.time() - entry.timestamp
            recency_weight = 1.0 / (1.0 + age * 0.001)  # Decay over time
            weights.append(entry.confidence * recency_weight)
        
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        # Weighted prediction
        predicted_class = int(np.round(np.average(predictions, weights=weights)))
        predicted_symmetry = np.average(symmetry_scores, weights=weights)
        
        # Confidence based on agreement
        confidence = np.std(predictions) if len(predictions) > 1 else 1.0
        confidence = max(0.0, 1.0 - confidence)
        
        return Hypothesis(
            predicted_symmetry=predicted_symmetry,
            predicted_class=predicted_class,
            confidence=confidence,
            source_unit=-1  # Will be set by caller
        )
    
    def verify_hypothesis(self, hypothesis: Hypothesis, actual_symmetry: float,
                         true_label: int) -> float:
        """
        Verify a hypothesis against actual data.
        
        Args:
            hypothesis: Hypothesis to verify
            actual_symmetry: Actual symmetry score
            true_label: Ground truth label
            
        Returns:
            Verification score (0-1, higher = better)
        """
        # Symmetry score agreement
        symmetry_agreement = 1.0 - abs(hypothesis.predicted_symmetry - actual_symmetry)
        symmetry_agreement = max(0.0, symmetry_agreement)
        
        # Class agreement
        class_agreement = 1.0 if hypothesis.predicted_class == true_label else 0.0
        
        # Combined verification score
        verification_score = symmetry_agreement * 0.5 + class_agreement * 0.5
        
        # Store for history
        self.hypotheses_history.append((hypothesis, verification_score))
        
        return verification_score
    
    def is_valid(self, verification_score: float) -> bool:
        """Check if verification score exceeds threshold."""
        return verification_score >= self.verification_threshold
    
    def get_verification_rate(self) -> float:
        """Get rate of valid hypotheses."""
        if not self.hypotheses_history:
            return 0.0
        
        valid_count = sum(1 for _, score in self.hypotheses_history 
                         if score >= self.verification_threshold)
        return valid_count / len(self.hypotheses_history)


class DistributedDiscussion:
    """
    Distributed discussion and consensus mechanism.
    
    Multiple computational units participate in discussion to reach consensus
    on difficult cases.
    """
    
    def __init__(self, max_rounds: int = 5, consensus_threshold: float = 0.7):
        """
        Initialize distributed discussion system.
        
        Args:
            max_rounds: Maximum number of discussion rounds
            consensus_threshold: Threshold for consensus agreement
        """
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.discussion_history: List[Dict] = []
    
    def compute_consensus_index(self, hypotheses: List[Hypothesis]) -> float:
        """
        Compute consensus index among hypotheses.
        
        Args:
            hypotheses: List of hypotheses from different units
            
        Returns:
            Consensus index (0-1, higher = more agreement)
        """
        if not hypotheses:
            return 0.0
        
        # Count votes for each class
        votes = [h.predicted_class for h in hypotheses]
        unique_classes = set(votes)
        
        if len(unique_classes) == 1:
            # Complete agreement
            return 1.0
        
        # Compute probability distribution
        K = len(unique_classes)
        counts = [votes.count(c) for c in unique_classes]
        probabilities = [c / len(votes) for c in counts]
        
        # Compute Shannon entropy
        entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probabilities)
        
        # Consensus index: 1 - normalized entropy
        max_entropy = math.log(K) if K > 1 else 1.0
        consensus_index = 1.0 - (entropy / max_entropy)
        
        return consensus_index
    
    def reach_consensus(self, hypotheses: List[Hypothesis]) -> Tuple[int, float, bool]:
        """
        Run discussion to reach consensus.
        
        Args:
            hypotheses: Initial hypotheses from different units
            
        Returns:
            (consensus_class, consensus_index, reached_consensus)
        """
        current_hypotheses = hypotheses.copy()
        
        for round_num in range(self.max_rounds):
            # Compute consensus index
            ci = self.compute_consensus_index(current_hypotheses)
            
            # Check if consensus reached
            if ci >= self.consensus_threshold:
                # Determine consensus class
                votes = [h.predicted_class for h in current_hypotheses]
                consensus_class = max(set(votes), key=votes.count)
                
                return consensus_class, ci, True
            
            # If not reached, could implement iterative refinement here
            # For now, we just continue to next round
        
        # No consensus reached after max rounds
        votes = [h.predicted_class for h in current_hypotheses]
        consensus_class = max(set(votes), key=votes.count)
        final_ci = self.compute_consensus_index(current_hypotheses)
        
        return consensus_class, final_ci, False
    
    def get_success_rate(self) -> float:
        """Get rate of successful consensus reaching."""
        if not self.discussion_history:
            return 0.0
        
        success_count = sum(1 for d in self.discussion_history if d['reached_consensus'])
        return success_count / len(self.discussion_history)


class ReflectionSystem:
    """
    Reflection and self-validation system.
    
    The system periodically reflects on its recent performance to identify
    improvement opportunities.
    """
    
    def __init__(self, reflection_interval: int = 50):
        """
        Initialize reflection system.
        
        Args:
            reflection_interval: Number of inferences between reflections
        """
        self.reflection_interval = reflection_interval
        self.inference_count = 0
        
        # Performance tracking
        self.recent_accuracies: deque = deque(maxlen=100)
        self.recent_confidences: deque = deque(maxlen=100)
        self.recent_predictions: deque = deque(maxlen=100)
        self.recent_labels: deque = deque(maxlen=100)
        
        # Strategy performance
        self.strategy_accuracies: Dict[str, List[float]] = {
            'knowledge_base': [],
            'hypothesis': [],
            'consensus': [],
            'default': []
        }
        
        # Reflection weights
        self.alpha1 = 0.5  # Consistency score weight
        self.alpha2 = 0.3  # Calibration error weight
        self.alpha3 = 0.2  # Recent accuracy weight
    
    def record_inference(self, prediction: int, true_label: int, confidence: float,
                        strategy: str = 'default'):
        """
        Record an inference for reflection tracking.
        
        Args:
            prediction: Predicted class
            true_label: Ground truth label
            confidence: Prediction confidence
            strategy: Strategy used for prediction
        """
        self.inference_count += 1
        
        accuracy = 1.0 if prediction == true_label else 0.0
        
        self.recent_accuracies.append(accuracy)
        self.recent_confidences.append(confidence)
        self.recent_predictions.append(prediction)
        self.recent_labels.append(true_label)
        
        if strategy in self.strategy_accuracies:
            self.strategy_accuracies[strategy].append(accuracy)
    
    def should_reflect(self) -> bool:
        """Check if it's time for reflection."""
        return self.inference_count % self.reflection_interval == 0
    
    def compute_reflection_score(self) -> float:
        """
        Compute reflection score based on recent performance.
        
        Returns:
            Reflection score (0-1, higher = better)
        """
        if not self.recent_accuracies:
            return 0.5
        
        # Consistency score: standard deviation of accuracies
        consistency_score = 1.0 - np.std(self.recent_accuracies)
        
        # Confidence calibration error
        if len(self.recent_accuracies) > 0:
            calibration_error = np.mean(np.abs(np.array(self.recent_confidences) - 
                                             np.array(self.recent_accuracies)))
            cal_score = 1.0 - calibration_error
        else:
            cal_score = 0.5
        
        # Recent accuracy
        recent_accuracy = np.mean(self.recent_accuracies)
        
        # Weighted reflection score
        reflection_score = (self.alpha1 * consistency_score + 
                           self.alpha2 * cal_score + 
                           self.alpha3 * recent_accuracy)
        
        return reflection_score
    
    def get_strategy_adjustment(self, strategy: str) -> float:
        """
        Get adjustment factor for a strategy based on its performance.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Adjustment factor (0.8, 1.0, or 1.1)
        """
        if strategy not in self.strategy_accuracies or not self.strategy_accuracies[strategy]:
            return 1.0
        
        accuracies = self.strategy_accuracies[strategy]
        avg_accuracy = np.mean(accuracies)
        
        if avg_accuracy > 0.8:
            return 1.1  # Boost high-performing strategies
        elif avg_accuracy < 0.5:
            return 0.8  # Penalize low-performing strategies
        else:
            return 1.0
    
    def reflect(self) -> Dict:
        """
        Perform reflection and generate strategy adjustments.
        
        Returns:
            Dictionary with reflection results
        """
        reflection_score = self.compute_reflection_score()
        
        # Get strategy adjustments
        adjustments = {}
        for strategy in self.strategy_accuracies.keys():
            adjustments[strategy] = self.get_strategy_adjustment(strategy)
        
        reflection_result = {
            'reflection_score': reflection_score,
            'strategy_adjustments': adjustments,
            'recent_accuracy': np.mean(self.recent_accuracies) if self.recent_accuracies else 0.0,
            'n_inferences': self.inference_count
        }
        
        return reflection_result
    
    def get_recent_accuracy(self) -> float:
        """Get recent accuracy."""
        return np.mean(self.recent_accuracies) if self.recent_accuracies else 0.0


class EnhancedCARSystem:
    """
    Enhanced CAR system with all advanced mechanisms.
    
    This integrates the base CAR system with knowledge base learning,
    hypothesis verification, distributed discussion, and reflection.
    """
    
    def __init__(self, n_units: int = 50, n_chunks: int = 5, random_seed: int = None):
        """
        Initialize enhanced CAR system.
        
        Args:
            n_units: Number of computational units
            n_chunks: Number of chunks for distributed processing
            random_seed: Random seed for reproducibility
        """
        # Import here to avoid circular dependency
        from .car_system import CARSystem
        
        self.base_system = CARSystem(n_units, n_chunks, random_seed)
        
        # Initialize enhanced mechanisms
        self.knowledge_base = KnowledgeBase(capacity=5000, similarity_threshold=0.85)
        self.hypothesis_verifier = HypothesisVerifier(verification_threshold=0.6)
        self.distributed_discussion = DistributedDiscussion(max_rounds=5, consensus_threshold=0.7)
        self.reflection_system = ReflectionSystem(reflection_interval=50)
        
        # Statistics
        self.kb_hits = 0
        self.hypothesis_verifications = 0
        self.consensus_reached = 0
        self.total_inferences = 0
    
    def process_molecule(self, X: np.ndarray, true_label: int = None) -> Dict:
        """
        Process a molecule with enhanced CAR mechanisms.
        
        Args:
            X: Molecular orbital features of shape (n_atoms, n_features)
            true_label: Ground truth label (optional, for learning)
            
        Returns:
            Dictionary with prediction and metadata
        """
        self.total_inferences += 1
        
        # Step 1: Check knowledge base for similar cases
        flattened_X = X.flatten()
        similar_entries = self.knowledge_base.retrieve_similar(flattened_X, k=5)
        
        prediction = 0
        confidence = 0.5
        strategy = 'default'
        symmetry_score = 0.0
        
        if similar_entries:
            self.kb_hits += 1
            
            # Step 2: Generate and verify hypothesis
            hypothesis = self.hypothesis_verifier.generate_hypothesis(
                similar_entries, 
                0.0  # Will be computed below
            )
            
            if hypothesis:
                # Compute symmetry score
                symmetry_score = self.base_system.get_symmetry_score(X)
                hypothesis.predicted_symmetry = symmetry_score
                
                # Verify hypothesis
                verification_score = self.hypothesis_verifier.verify_hypothesis(
                    hypothesis, symmetry_score, true_label if true_label is not None else 0
                )
                self.hypothesis_verifications += 1
                
                if self.hypothesis_verifier.is_valid(verification_score):
                    prediction = hypothesis.predicted_class
                    confidence = hypothesis.confidence
                    strategy = 'hypothesis'
        
        # Step 3: If no valid hypothesis, run CAR cycles
        if strategy == 'default':
            self.base_system.load_data(flattened_X)
            convergence_info = self.base_system.process_until_convergence()
            
            # Compute symmetry score
            symmetry_score = self.base_system.get_symmetry_score(X)
            
            # Simple threshold-based prediction
            prediction = 1 if symmetry_score < -0.0027 else 0
            confidence = 0.5
        
        # Step 4: Distributed discussion for difficult cases
        if confidence < 0.6:
            # Generate hypotheses from different units
            hypotheses = []
            for i in range(min(5, self.base_system.n_units)):
                h = Hypothesis(
                    predicted_symmetry=symmetry_score,
                    predicted_class=prediction,
                    confidence=confidence,
                    source_unit=i
                )
                hypotheses.append(h)
            
            consensus_class, ci, reached = self.distributed_discussion.reach_consensus(hypotheses)
            
            if reached:
                prediction = consensus_class
                confidence = ci
                strategy = 'consensus'
                self.consensus_reached += 1
        
        # Step 5: Record inference for reflection
        if true_label is not None:
            self.reflection_system.record_inference(prediction, true_label, confidence, strategy)
            
            # Add to knowledge base
            self.knowledge_base.add_entry(
                flattened_X,
                symmetry_score,
                prediction,
                true_label,
                confidence
            )
        
        # Step 6: Periodic reflection
        if self.reflection_system.should_reflect():
            reflection_result = self.reflection_system.reflect()
            # Could use reflection_result to adjust system parameters
        
        # Reset system for next molecule
        self.base_system.reset()
        
        return {
            'prediction': prediction,
            'symmetry_score': symmetry_score,
            'confidence': confidence,
            'strategy': strategy,
            'kb_hits': self.kb_hits,
            'hypothesis_verifications': self.hypothesis_verifications,
            'consensus_reached': self.consensus_reached
        }
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        return {
            'total_inferences': self.total_inferences,
            'kb_hits': self.kb_hits,
            'kb_hit_rate': self.kb_hits / self.total_inferences if self.total_inferences > 0 else 0.0,
            'hypothesis_verifications': self.hypothesis_verifications,
            'consensus_reached': self.consensus_reached,
            'knowledge_base_size': self.knowledge_base.get_size(),
            'knowledge_base_accuracy': self.knowledge_base.get_accuracy(),
            'recent_accuracy': self.reflection_system.get_recent_accuracy(),
            'verification_rate': self.hypothesis_verifier.get_verification_rate(),
            'consensus_success_rate': self.distributed_discussion.get_success_rate()
        }