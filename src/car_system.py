"""
CAR (Compare-Adjust-Record) System Implementation
Based on the paper: Emergent Pattern Detection Through Iterative Computational Unit Interactions with Bounded State Dynamics

This module implements the complete CAR computational architecture for pattern detection
without gradient-based optimization or backpropagation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class UnitState:
    """State triplet for each computational unit."""
    A: float  # Activation weight
    v: float  # Validation score
    x: np.ndarray  # Data sample
    history: List[Tuple] = field(default_factory=list)  # Interaction history
    
    def __post_init__(self):
        if self.A < 0.1:
            self.A = 0.1
        if self.A > 0.9:
            self.A = 0.9
        if self.v < 0:
            self.v = 0
        if self.v > 1:
            self.v = 1


class CARSystem:
    """
    Complete CAR (Compare-Adjust-Record) System implementation.
    
    This system implements autonomous computational units that maintain independent
    state representations and interact through CAR cycles without gradient-based
    optimization or backpropagation.
    """
    
    def __init__(self, n_units: int = 50, n_chunks: int = 5, random_seed: int = None):
        """
        Initialize the CAR system.
        
        Args:
            n_units: Number of computational units
            n_chunks: Number of chunks for distributed processing
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.n_units = n_units
        self.n_chunks = n_chunks
        
        # Parameters from paper
        self.alpha = 0.5  # Tanh input scale
        self.beta = 0.25  # Consensus learning rate
        self.gamma = 0.4  # Diversity threshold
        self.epsilon = 1e-10  # Numerical stability constant
        self.theta_sat = 0.6  # Activation threshold
        self.delta_omega = 0.1  # Communication weight offset
        self.gamma_v = 0.1  # Validation learning rate
        self.gamma_w = 0.1  # Trust update learning rate
        
        # Convergence parameters
        self.theta_c = 0.06  # Convergence threshold
        self.tau_min = 30  # Minimum iterations before checking
        self.tau_max = 400  # Maximum total iterations
        
        # Initialize chunk assignments (before trust weights)
        self.chunk_assignments = np.random.randint(0, n_chunks, n_units)
        
        # Initialize units
        self.units: List[UnitState] = []
        for i in range(n_units):
            self.units.append(UnitState(
                A=0.1,
                v=0.5,
                x=np.array([]),
                history=[]
            ))
        
        # Initialize trust weights
        self.trust_weights = np.full((n_units, n_units), 0.3)
        self._initialize_trust_weights()
        
        # Initialize chunk representatives
        self.chunk_representatives = {}
        self._update_chunk_representatives()
        
        # Track convergence
        self.initial_variance = 0.0
        self.converged = False
        self.n_iterations = 0
        
        # Track activation history for adaptive learning
        self.delta_A_history = deque(maxlen=10)
    
    def _initialize_trust_weights(self):
        """Initialize trust weights based on chunk assignments."""
        for i in range(self.n_units):
            for j in range(self.n_units):
                if self.chunk_assignments[i] == self.chunk_assignments[j]:
                    self.trust_weights[i, j] = 0.5
                else:
                    self.trust_weights[i, j] = 0.3
    
    def _update_chunk_representatives(self):
        """Update chunk representatives based on highest activation."""
        for chunk_id in range(self.n_chunks):
            chunk_units = [i for i in range(self.n_units) 
                          if self.chunk_assignments[i] == chunk_id]
            if chunk_units:
                rep_idx = max(chunk_units, key=lambda i: self.units[i].A)
                self.chunk_representatives[chunk_id] = rep_idx
    
    def _tanh_activation(self, x: np.ndarray, A: float) -> Tuple[float, bool]:
        """
        Compute tanh activation and detect high activation.
        
        Args:
            x: Input data vector
            A: Activation weight
            
        Returns:
            (tanh_output, is_high_activation)
        """
        tanh_input = np.linalg.norm(x) * np.sqrt(A) * self.alpha
        tanh_output = np.tanh(tanh_input)
        is_high = abs(tanh_output) > self.theta_sat
        return tanh_output, is_high
    
    def _compute_similarity(self, x_i: np.ndarray, x_j: np.ndarray) -> float:
        """Compute cosine similarity between two data vectors."""
        norm_i = np.linalg.norm(x_i)
        norm_j = np.linalg.norm(x_j)
        return np.dot(x_i, x_j) / (norm_i * norm_j + self.epsilon)
    
    def _get_peers(self, unit_idx: int) -> List[int]:
        """Get peer units for a given unit (same chunk)."""
        chunk_id = self.chunk_assignments[unit_idx]
        return [i for i in range(self.n_units) 
                if i != unit_idx and self.chunk_assignments[i] == chunk_id]
    
    def _consensus_update(self, unit_idx: int, peers: List[int]) -> float:
        """
        Perform consensus update for a unit.
        
        Args:
            unit_idx: Index of the unit to update
            peers: List of peer unit indices
            
        Returns:
            Updated activation weight
        """
        unit = self.units[unit_idx]
        
        # Compute weighted peer influence
        peer_influence = 0.0
        total_weight = 0.0
        
        for peer_idx in peers:
            peer = self.units[peer_idx]
            weight = abs(np.tanh(peer.A)) + self.delta_omega
            peer_influence += weight * peer.A
            total_weight += weight
        
        if total_weight > 0:
            peer_influence /= total_weight
        
        # Update activation weight
        new_A = (1 - self.beta) * unit.A + self.beta * peer_influence
        
        # Clip to bounds
        new_A = np.clip(new_A, 0.1, 0.9)
        
        # Store delta for adaptive learning
        self.delta_A_history.append(new_A - unit.A)
        
        return new_A
    
    def _diversity_update(self, unit_idx: int, peers: List[int]) -> Optional[float]:
        """
        Apply diversity-preserving update if needed.
        
        Args:
            unit_idx: Index of the unit
            peers: List of peer unit indices
            
        Returns:
            Updated activation weight or None if no update needed
        """
        unit = self.units[unit_idx]
        diversity_check = False
        
        for peer_idx in peers:
            peer = self.units[peer_idx]
            if abs(np.tanh(peer.A) - np.tanh(unit.A)) > self.gamma:
                diversity_check = True
                break
        
        if diversity_check:
            xi = np.random.uniform(-0.5, 0.5)
            new_A = unit.A + 0.015 * xi
            new_A = np.clip(new_A, 0.1, 0.9)
            return new_A
        
        return None
    
    def _check_convergence(self) -> Tuple[bool, float, float]:
        """
        Check if the system has converged.
        
        Returns:
            (is_converged, current_std, variance_reduction_ratio)
        """
        if self.n_iterations < self.tau_min:
            return False, 0.0, 0.0
        
        # Compute current variance
        activation_weights = [unit.A for unit in self.units]
        current_std = np.std(activation_weights)
        
        # Compute variance reduction ratio
        if self.initial_variance > 0:
            rho = 1 - current_std / self.initial_variance
        else:
            rho = 0.0
        
        # Check convergence criteria
        converged = current_std < self.theta_c and rho > 0.7
        
        return converged, current_std, rho
    
    def load_data(self, X: np.ndarray):
        """
        Load data into all units.
        
        Args:
            X: Data matrix of shape (n_features,) or (n_units, n_features)
        """
        if X.ndim == 1:
            # Broadcast same data to all units
            for unit in self.units:
                unit.x = X.copy()
        elif X.ndim == 2 and X.shape[0] == self.n_units:
            # Different data for each unit
            for i, unit in enumerate(self.units):
                unit.x = X[i, :].copy()
        else:
            raise ValueError(f"Invalid data shape: {X.shape}")
    
    def run_car_cycle(self, unit_idx: int, peers: List[int]):
        """
        Execute a complete CAR cycle for a unit.
        
        Args:
            unit_idx: Index of the unit
            peers: List of peer unit indices
        """
        unit = self.units[unit_idx]
        
        # Compare Phase: Compute similarities
        similarities = {}
        for peer_idx in peers:
            similarities[peer_idx] = self._compute_similarity(unit.x, self.units[peer_idx].x)
        
        # Adjust Phase: Update activation through consensus
        old_A = unit.A
        new_A = self._consensus_update(unit_idx, peers)
        
        # Check for diversity update
        diversity_A = self._diversity_update(unit_idx, peers)
        if diversity_A is not None:
            new_A = diversity_A
        
        unit.A = new_A
        
        # Record Phase: Accumulate interaction history
        for peer_idx, phi in similarities.items():
            unit.history.append((peer_idx, phi, new_A - old_A))
    
    def process_until_convergence(self, max_iterations: int = None) -> Dict:
        """
        Run CAR cycles until convergence or max iterations.
        
        Args:
            max_iterations: Maximum iterations (default: self.tau_max)
            
        Returns:
            Dictionary with convergence information
        """
        if max_iterations is None:
            max_iterations = self.tau_max
        
        # Initialize variance tracking
        activation_weights = [unit.A for unit in self.units]
        self.initial_variance = np.var(activation_weights)
        
        for iteration in range(max_iterations):
            self.n_iterations = iteration + 1
            
            # Execute CAR cycle for all units
            for i in range(self.n_units):
                peers = self._get_peers(i)
                self.run_car_cycle(i, peers)
            
            # Update chunk representatives periodically
            if iteration % 10 == 0:
                self._update_chunk_representatives()
            
            # Check convergence
            converged, current_std, rho = self._check_convergence()
            
            if converged:
                self.converged = True
                break
        
        return {
            'converged': self.converged,
            'n_iterations': self.n_iterations,
            'final_std': current_std if self.converged else np.std([u.A for u in self.units]),
            'variance_reduction': rho if self.converged else 0.0
        }
    
    def get_symmetry_score(self, X: np.ndarray) -> float:
        """
        Compute symmetry score for molecular orbital features.
        
        Args:
            X: Molecular orbital features of shape (n_atoms, n_features)
            
        Returns:
            Symmetry score (lower = more symmetric)
        """
        n_atoms, n_features = X.shape
        K = min(n_atoms, n_features)
        
        symmetry_indicators = []
        
        for k in range(K):
            # Compute pairwise similarities for this orbital/atom pair
            similarities = []
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    sim = self._compute_similarity(X[i, :], X[j, :])
                    similarities.append(sim)
            
            if similarities:
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                indicator = mean_sim - 0.5 * std_sim
                symmetry_indicators.append(indicator)
        
        # Aggregate with decreasing weights
        if symmetry_indicators:
            weights = np.linspace(1.0, 0.5, len(symmetry_indicators))
            weights = weights / np.sum(weights)
            symmetry_score = np.sum(weights * np.array(symmetry_indicators))
        else:
            symmetry_score = 0.0
        
        return symmetry_score
    
    def reset(self):
        """Reset all units to initial state."""
        for unit in self.units:
            unit.A = 0.1
            unit.v = 0.5
            unit.x = np.array([])
            unit.history = []
        
        self.converged = False
        self.n_iterations = 0
        self.delta_A_history.clear()