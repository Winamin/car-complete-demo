"""
Experimental Pipeline with Statistical Analysis

This module implements the complete experimental pipeline for evaluating
the enhanced CAR mechanism on molecular symmetry detection using QM9 data.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime

from .enhanced_car import EnhancedCARSystem
from .qm9_dataset import QM9Dataset, MolecularSymmetryGenerator


class StatisticalAnalyzer:
    """
    Statistical analysis tools for evaluating experimental results.
    """
    
    @staticmethod
    def compute_z_score(mean1: float, mean2: float, std1: float, std2: float,
                       n1: int, n2: int) -> float:
        """
        Compute Z-score for comparing two means.
        
        Args:
            mean1, mean2: Sample means
            std1, std2: Sample standard deviations
            n1, n2: Sample sizes
            
        Returns:
            Z-score
        """
        se = np.sqrt(std1**2 / n1 + std2**2 / n2)
        if se == 0:
            return 0.0
        return (mean1 - mean2) / se
    
    @staticmethod
    def compute_cohens_d(mean1: float, mean2: float, std1: float, std2: float,
                        n1: int, n2: int) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            mean1, mean2: Sample means
            std1, std2: Sample standard deviations
            n1, n2: Sample sizes
            
        Returns:
            Cohen's d
        """
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def compute_discrimination_index(mean1: float, mean2: float, std1: float, std2: float,
                                    n1: int, n2: int) -> float:
        """
        Compute discrimination index.
        
        Args:
            mean1, mean2: Sample means
            std1, std2: Sample standard deviations
            n1, n2: Sample sizes
            
        Returns:
            Discrimination index
        """
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return abs(mean1 - mean2) / pooled_std
    
    @staticmethod
    def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, int]:
        """
        Find optimal threshold for classification using bidirectional search.
        
        Args:
            scores: Symmetry scores
            labels: True labels (0 or 1)
            
        Returns:
            (optimal_threshold, best_accuracy, direction)
            where direction: 1 if score > threshold predicts class 1, -1 if score < threshold predicts class 1
        """
        score_min = np.min(scores)
        score_max = np.max(scores)
        
        best_threshold = score_min
        best_accuracy = 0.0
        best_direction = 1
        
        # Scan 100 threshold values
        for k in range(100):
            threshold = score_min + k * (score_max - score_min) / 99
            
            # Direction 1: score > threshold predicts class 1
            predictions1 = (scores > threshold).astype(int)
            accuracy1 = np.mean(predictions1 == labels)
            
            # Direction 2: score < threshold predicts class 1
            predictions2 = (scores < threshold).astype(int)
            accuracy2 = np.mean(predictions2 == labels)
            
            if accuracy1 > best_accuracy:
                best_accuracy = accuracy1
                best_threshold = threshold
                best_direction = 1
            
            if accuracy2 > best_accuracy:
                best_accuracy = accuracy2
                best_threshold = threshold
                best_direction = -1
        
        return best_threshold, best_accuracy, best_direction
    
    @staticmethod
    def compute_precision_recall_f1(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute precision, recall, and F1-score.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Dictionary with precision, recall, f1, tp, fp, fn, tn
        """
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }


class ExperimentRunner:
    """
    Run complete experiments with the enhanced CAR mechanism.
    """
    
    def __init__(self, n_units: int = 50, n_chunks: int = 5, random_seed: int = None):
        """
        Initialize experiment runner.
        
        Args:
            n_units: Number of computational units
            n_chunks: Number of chunks
            random_seed: Random seed for reproducibility
        """
        self.n_units = n_units
        self.n_chunks = n_chunks
        self.random_seed = random_seed
        self.analyzer = StatisticalAnalyzer()
    
    def run_single_experiment(self, features: np.ndarray, labels: np.ndarray,
                              random_seed: int = None) -> Dict:
        """
        Run a single experiment on the dataset.
        
        Args:
            features: Molecular features of shape (n_molecules, n_atoms, n_features)
            labels: True labels (0 or 1)
            random_seed: Random seed for this experiment
            
        Returns:
            Dictionary with experiment results
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        print(f"\nRunning experiment with seed {random_seed}...")
        
        # Initialize enhanced CAR system
        car_system = EnhancedCARSystem(
            n_units=self.n_units,
            n_chunks=self.n_chunks,
            random_seed=random_seed
        )
        
        # Process all molecules
        n_molecules = features.shape[0]
        predictions = []
        symmetry_scores = []
        
        start_time = time.time()
        
        for i in range(n_molecules):
            result = car_system.process_molecule(features[i], labels[i])
            predictions.append(result['prediction'])
            symmetry_scores.append(result['symmetry_score'])
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_molecules} molecules...")
        
        elapsed_time = time.time() - start_time
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        symmetry_scores = np.array(symmetry_scores)
        
        # Separate scores by class
        high_symmetry_scores = symmetry_scores[labels == 1]
        low_symmetry_scores = symmetry_scores[labels == 0]
        
        # Compute discrimination metrics
        mean_high = np.mean(high_symmetry_scores)
        mean_low = np.mean(low_symmetry_scores)
        std_high = np.std(high_symmetry_scores)
        std_low = np.std(low_symmetry_scores)
        n_high = len(high_symmetry_scores)
        n_low = len(low_symmetry_scores)
        
        z_score = self.analyzer.compute_z_score(mean_high, mean_low, std_high, std_low, n_high, n_low)
        cohens_d = self.analyzer.compute_cohens_d(mean_high, mean_low, std_high, std_low, n_high, n_low)
        discrimination_index = self.analyzer.compute_discrimination_index(mean_high, mean_low, std_high, std_low, n_high, n_low)
        
        # Find optimal threshold
        optimal_threshold, threshold_accuracy, direction = self.analyzer.find_optimal_threshold(
            symmetry_scores, labels
        )
        
        # Make predictions using optimal threshold
        if direction == 1:
            final_predictions = (symmetry_scores > optimal_threshold).astype(int)
        else:
            final_predictions = (symmetry_scores < optimal_threshold).astype(int)
        
        # Compute classification metrics
        prf1_metrics = self.analyzer.compute_precision_recall_f1(final_predictions, labels)
        
        # Get system statistics
        system_stats = car_system.get_statistics()
        
        results = {
            'random_seed': random_seed,
            'elapsed_time': elapsed_time,
            'mean_high_symmetry': mean_high,
            'mean_low_symmetry': mean_low,
            'std_high_symmetry': std_high,
            'std_low_symmetry': std_low,
            'n_high_symmetry': n_high,
            'n_low_symmetry': n_low,
            'z_score': z_score,
            'cohens_d': cohens_d,
            'discrimination_index': discrimination_index,
            'optimal_threshold': optimal_threshold,
            'threshold_accuracy': threshold_accuracy,
            'prediction_direction': direction,
            'accuracy': np.mean(final_predictions == labels),
            'precision': prf1_metrics['precision'],
            'recall': prf1_metrics['recall'],
            'f1': prf1_metrics['f1'],
            'tp': prf1_metrics['tp'],
            'fp': prf1_metrics['fp'],
            'fn': prf1_metrics['fn'],
            'tn': prf1_metrics['tn'],
            'system_stats': system_stats
        }
        
        print(f"\nExperiment completed in {elapsed_time:.2f} seconds")
        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"  F1-score: {results['f1']:.2%}")
        print(f"  Precision: {results['precision']:.2%}")
        print(f"  Recall: {results['recall']:.2%}")
        print(f"  Z-score: {z_score:.2f}")
        print(f"  Cohen's d: {cohens_d:.2f}")
        
        return results
    
    def run_multi_experiment(self, n_experiments: int = 10, n_molecules: int = 2000,
                            high_symmetry_ratio: float = 0.25, n_atoms: int = 9,
                            n_features: int = 5, base_seed: int = 42) -> Dict:
        """
        Run multiple independent experiments with random data generation.
        
        Args:
            n_experiments: Number of independent experiments
            n_molecules: Total molecules per experiment
            high_symmetry_ratio: Ratio of high-symmetry molecules
            n_atoms: Number of atoms per molecule
            n_features: Number of orbital features per atom
            base_seed: Base random seed
            
        Returns:
            Dictionary with aggregated results across all experiments
        """
        print(f"\n{'='*60}")
        print(f"Running {n_experiments} independent experiments")
        print(f"{'='*60}")
        
        # Initialize QM9 dataset handler to get statistics
        qm9 = QM9Dataset()
        qm9_stats = qm9.get_qm9_statistics()
        
        # Initialize symmetry generator
        symmetry_gen = MolecularSymmetryGenerator(qm9_stats)
        
        # Store results from all experiments
        all_results = []
        
        for exp_idx in range(n_experiments):
            seed = base_seed + exp_idx
            
            # Generate dataset with this seed
            features, labels = symmetry_gen.generate_complete_dataset(
                n_total=n_molecules,
                high_symmetry_ratio=high_symmetry_ratio,
                n_atoms=n_atoms,
                n_features=n_features,
                random_seed=seed
            )
            
            # Run experiment
            results = self.run_single_experiment(features, labels, seed)
            all_results.append(results)
        
        # Compute aggregated statistics
        aggregated = self._aggregate_results(all_results)
        
        print(f"\n{'='*60}")
        print(f"Aggregated Results Across {n_experiments} Experiments")
        print(f"{'='*60}")
        print(f"  Mean Accuracy: {aggregated['mean_accuracy']:.2%} ± {aggregated['std_accuracy']:.2%}")
        print(f"  Mean F1-score: {aggregated['mean_f1']:.2%} ± {aggregated['std_f1']:.2%}")
        print(f"  Mean Precision: {aggregated['mean_precision']:.2%} ± {aggregated['std_precision']:.2%}")
        print(f"  Mean Recall: {aggregated['mean_recall']:.2%} ± {aggregated['std_recall']:.2%}")
        print(f"  Mean Z-score: {aggregated['mean_z_score']:.2f} ± {aggregated['std_z_score']:.2f}")
        print(f"  Mean Cohen's d: {aggregated['mean_cohens_d']:.2f} ± {aggregated['std_cohens_d']:.2f}")
        print(f"  Total time: {aggregated['total_time']:.2f} seconds")
        print(f"  Mean time per experiment: {aggregated['mean_time']:.2f} seconds")
        print(f"{'='*60}\n")
        
        return aggregated
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """
        Aggregate results from multiple experiments.
        
        Args:
            all_results: List of result dictionaries from individual experiments
            
        Returns:
            Dictionary with aggregated statistics
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'z_score', 'cohens_d',
                  'discrimination_index', 'optimal_threshold', 'elapsed_time']
        
        aggregated = {}
        
        for metric in metrics:
            values = [r[metric] for r in all_results]
            aggregated[f'mean_{metric}'] = np.mean(values)
            aggregated[f'std_{metric}'] = np.std(values)
            aggregated[f'min_{metric}'] = np.min(values)
            aggregated[f'max_{metric}'] = np.max(values)
        
        # Total time
        aggregated['total_time'] = sum(r['elapsed_time'] for r in all_results)
        aggregated['mean_time'] = np.mean([r['elapsed_time'] for r in all_results])
        
        # Store all individual results
        aggregated['all_results'] = all_results
        
        return aggregated
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            filename: Output filename (default: auto-generated with timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"car_experiment_results_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {filepath}")


def main():
    """
    Main function to run the complete experimental pipeline.
    """
    print("\n" + "="*60)
    print("Enhanced CAR Mechanism Experimental Pipeline")
    print("Molecular Symmetry Detection with QM9 Data")
    print("="*60)
    
    # Configuration
    n_experiments = 10
    n_molecules = 2000
    high_symmetry_ratio = 0.25
    n_atoms = 9
    n_features = 5
    n_units = 50
    n_chunks = 5
    base_seed = 42
    
    # Create experiment runner
    runner = ExperimentRunner(
        n_units=n_units,
        n_chunks=n_chunks,
        random_seed=base_seed
    )
    
    # Run multi-experiment evaluation
    results = runner.run_multi_experiment(
        n_experiments=n_experiments,
        n_molecules=n_molecules,
        high_symmetry_ratio=high_symmetry_ratio,
        n_atoms=n_atoms,
        n_features=n_features,
        base_seed=base_seed
    )
    
    # Save results
    runner.save_results(results)
    
    print("\n" + "="*60)
    print("Experimental Pipeline Completed Successfully!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()