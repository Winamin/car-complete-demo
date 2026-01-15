"""
QM9 Dataset Download and Processing Module

This module handles downloading and processing the real QM9 dataset to extract
authentic molecular statistical properties for experimental validation.
"""

import numpy as np
import os
import requests
from typing import Dict, Tuple
import gzip
import pickle


class QM9Dataset:
    """
    QM9 Dataset handler for downloading and processing real molecular data.
    
    The QM9 dataset contains computed geometric, energetic, electronic, and
    thermodynamic properties for approximately 130,000 stable small organic
    molecules composed of C, H, O, N, and F elements.
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize QM9 dataset handler.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, 'qm9.pkl.gz')
        self.metadata = {
            'total_molecules': 133885,
            'mean_homo_lumo_gap': 0.2511,  # eV
            'std_homo_lumo_gap': 0.0475,   # eV
            'min_homo_lumo_gap': 0.0246,   # eV
            'max_homo_lumo_gap': 0.6221,   # eV
            'elements': ['C', 'H', 'O', 'N', 'F'],
            'avg_atoms_per_molecule': 9
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def download_dataset(self, url: str = None) -> str:
        """
        Download QM9 dataset from available sources.
        
        Args:
            url: URL to download from (default: tries multiple sources)
            
        Returns:
            Path to downloaded file or None if download fails
        """
        # List of alternative download sources
        alternative_urls = [
            # Primary: DeepChem S3
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.pkl.gz",
            # Alternative 1: Figshare (if available)
            "https://figshare.com/ndownloader/files/3195389",
            # Alternative 2: Zenodo mirror (if available)
            "https://zenodo.org/record/1343515/files/qm9_data.npz",
            # Alternative 3: GitHub releases
            "https://github.com/aspuru-guzik-group/generative_models/raw/master/data/qm9.pkl.gz",
        ]
        
        if url is None:
            urls_to_try = alternative_urls
        else:
            urls_to_try = [url]
        
        for url in urls_to_try:
            print(f"\nAttempting to download QM9 dataset from:")
            print(f"  {url}")
            print(f"This may take a while as the dataset is large (~500MB)...")
            
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(self.dataset_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownloaded: {percent:.1f}%", end='')
                
                print(f"\nDataset downloaded successfully to {self.dataset_path}")
                return self.dataset_path
                
            except requests.RequestException as e:
                print(f"\nError downloading from this URL: {e}")
                continue  # Try next URL
        
        print("\nAll download sources failed or are unavailable.")
        print("Using pre-defined QM9 statistics instead.")
        return None
    
    def load_dataset(self) -> Tuple[np.ndarray, Dict]:
        """
        Load QM9 dataset from file.
        
        Returns:
            (molecules_data, properties_dict)
        """
        if not os.path.exists(self.dataset_path):
            print("Dataset file not found. Downloading...")
            self.download_dataset()
        
        if os.path.exists(self.dataset_path):
            print(f"Loading QM9 dataset from {self.dataset_path}...")
            
            with gzip.open(self.dataset_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Loaded {len(data)} molecules from QM9 dataset")
            return data, self.metadata
        else:
            print("Using pre-defined QM9 statistics for feature generation")
            return None, self.metadata
    
    def load_dataset_from_csv(self) -> Dict[str, float]:
        """
        Load QM9 dataset from CSV file and extract statistics.
        
        Returns:
            Dictionary with statistical properties
        """
        csv_path = os.path.join(self.data_dir, 'gdb9.sdf.csv')
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}")
            return None
        
        print(f"Loading QM9 dataset from CSV: {csv_path}")
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Extract HOMO-LUMO gap statistics
            gap = df['gap'].astype(float)
            
            stats = {
                'total_molecules': len(df),
                'mean': gap.mean(),
                'std': gap.std(),
                'min': gap.min(),
                'max': gap.max(),
                'median': gap.median(),
                'q25': gap.quantile(0.25),
                'q75': gap.quantile(0.75)
            }
            
            print(f"\nHOMO-LUMO Gap Statistics from Real QM9 Data:")
            print(f"  Total molecules: {stats['total_molecules']}")
            print(f"  Mean: {stats['mean']:.4f} eV")
            print(f"  Standard deviation: {stats['std']:.4f} eV")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}] eV")
            print(f"  Median: {stats['median']:.4f} eV")
            
            return stats
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def extract_homo_lumo_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract HOMO-LUMO gap statistics from QM9 data.
        
        Args:
            data: QM9 dataset array
            
        Returns:
            Dictionary with statistical properties
        """
        # QM9 dataset structure: each molecule has properties including HOMO-LUMO gap
        # The HOMO-LUMO gap is typically at index 3 in the property array
        
        homo_lumo_gaps = []
        
        for mol in data:
            # Extract HOMO-LUMO gap (property index 3 in standard QM9)
            if len(mol) > 3:
                gap = mol[3]
                homo_lumo_gaps.append(gap)
        
        homo_lumo_gaps = np.array(homo_lumo_gaps)
        
        stats = {
            'total_molecules': len(homo_lumo_gaps),
            'mean': np.mean(homo_lumo_gaps),
            'std': np.std(homo_lumo_gaps),
            'min': np.min(homo_lumo_gaps),
            'max': np.max(homo_lumo_gaps),
            'median': np.median(homo_lumo_gaps),
            'q25': np.percentile(homo_lumo_gaps, 25),
            'q75': np.percentile(homo_lumo_gaps, 75)
        }
        
        print("\nHOMO-LUMO Gap Statistics from Real QM9 Data:")
        print(f"  Total molecules: {stats['total_molecules']}")
        print(f"  Mean: {stats['mean']:.4f} eV")
        print(f"  Standard deviation: {stats['std']:.4f} eV")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}] eV")
        print(f"  Median: {stats['median']:.4f} eV")
        
        return stats
    
    def get_qm9_statistics(self) -> Dict[str, float]:
        """
        Get QM9 statistics (either from real data or pre-defined values).
        
        Returns:
            Dictionary with QM9 statistical properties
        """
        # Priority 1: Try to load from CSV file (real data)
        csv_stats = self.load_dataset_from_csv()
        if csv_stats is not None:
            return csv_stats
        
        # Priority 2: Try to load from pickle file
        data, _ = self.load_dataset()
        if data is not None:
            stats = self.extract_homo_lumo_statistics(data)
            return stats
        
        # Priority 3: Return pre-defined statistics from paper
        print("\nUsing Pre-defined QM9 Statistics from Paper:")
        print(f"  Total molecules: {self.metadata['total_molecules']}")
        print(f"  Mean HOMO-LUMO gap: {self.metadata['mean_homo_lumo_gap']:.4f} eV")
        print(f"  Standard deviation: {self.metadata['std_homo_lumo_gap']:.4f} eV")
        print(f"  Range: [{self.metadata['min_homo_lumo_gap']:.4f}, {self.metadata['max_homo_lumo_gap']:.4f}] eV")
        
        return {
                'total_molecules': self.metadata['total_molecules'],
                'mean': self.metadata['mean_homo_lumo_gap'],
                'std': self.metadata['std_homo_lumo_gap'],
                'min': self.metadata['min_homo_lumo_gap'],
                'max': self.metadata['max_homo_lumo_gap']
            }
    
    def generate_qm9_based_features(self, n_molecules: int, n_atoms: int = 9, 
                                    n_features: int = 5, random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate molecular features based on QM9 statistics.
        
        Args:
            n_molecules: Number of molecules to generate
            n_atoms: Number of atoms per molecule (default: 9 from QM9)
            n_features: Number of orbital features per atom
            random_seed: Random seed for reproducibility
            
        Returns:
            (features, labels) where features shape is (n_molecules, n_atoms, n_features)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Get QM9 statistics
        stats = self.get_qm9_statistics()
        
        # Generate features with QM9-based distribution
        features = np.random.normal(
            loc=stats['mean'],
            scale=stats['std'] * 0.5,
            size=(n_molecules, n_atoms, n_features)
        )
        
        # Add noise to simulate real molecular variability
        noise = np.random.normal(
            loc=0,
            scale=stats['std'] * 0.08,
            size=(n_molecules, n_atoms, n_features)
        )
        features += noise
        
        # Generate random labels (will be overridden by symmetry generation)
        labels = np.zeros(n_molecules, dtype=int)
        
        return features, labels


class MolecularSymmetryGenerator:
    """
    Generate molecular features with controlled symmetry properties.
    
    This class generates high-symmetry and low-symmetry molecular features
    based on real QM9 statistics and point group symmetries.
    """
    
    def __init__(self, qm9_stats: Dict[str, float]):
        """
        Initialize symmetry generator.
        
        Args:
            qm9_stats: QM9 statistical properties
        """
        self.qm9_stats = qm9_stats
        self.symmetry_types = ['C2v', 'C2h', 'Cs', 'Ci', 'Cn']
        self.symmetry_probabilities = [0.30, 0.20, 0.20, 0.15, 0.15]
    
    def generate_high_symmetry_features(self, n_molecules: int, n_atoms: int = 9,
                                       n_features: int = 5, random_seed: int = None) -> np.ndarray:
        """
        Generate high-symmetry molecular features.
        
        Args:
            n_molecules: Number of high-symmetry molecules
            n_atoms: Number of atoms per molecule
            n_features: Number of orbital features per atom
            random_seed: Random seed for reproducibility
            
        Returns:
            Features array of shape (n_molecules, n_atoms, n_features)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        features = np.zeros((n_molecules, n_atoms, n_features))
        
        for i in range(n_molecules):
            # Select symmetry type
            symmetry_type = np.random.choice(
                self.symmetry_types,
                p=self.symmetry_probabilities
            )
            
            # Generate base features
            X = np.random.normal(
                loc=self.qm9_stats['mean'],
                scale=self.qm9_stats['std'] * 0.5,
                size=(n_atoms, n_features)
            )
            
            # Add noise
            X += np.random.normal(
                loc=0,
                scale=self.qm9_stats['std'] * 0.08,
                size=(n_atoms, n_features)
            )
            
            # Apply symmetry constraints
            if symmetry_type == 'C2v':
                # Water-like symmetry: 2-fold rotation + 2 vertical mirror planes
                for j in range(n_atoms // 2):
                    X[2*j+1, :] = X[2*j, :]
            elif symmetry_type == 'C2h':
                # Antisymmetric mirror planes with inversion center
                for j in range(n_atoms // 2):
                    X[2*j+1, :] = -X[2*j, :]
            elif symmetry_type == 'Cs':
                # Single mirror plane
                mid = n_atoms // 2
                for j in range(mid):
                    X[n_atoms - 1 - j, :] = X[j, :]
            elif symmetry_type == 'Ci':
                # Inversion center only
                for j in range(n_atoms // 2):
                    X[n_atoms - 1 - j, :] = -X[j, :]
            elif symmetry_type == 'Cn':
                # n-fold rotation axis (cyclic symmetry)
                n_rot = np.random.randint(3, 5)  # 3-fold or 4-fold
                for k in range(n_atoms // n_rot):
                    base_idx = k * n_rot
                    for r in range(1, n_rot):
                        if base_idx + r < n_atoms:
                            X[base_idx + r, :] = X[base_idx, :]
            
            features[i, :, :] = X
        
        return features
    
    def generate_low_symmetry_features(self, n_molecules: int, n_atoms: int = 9,
                                      n_features: int = 5, random_seed: int = None) -> np.ndarray:
        """
        Generate low-symmetry molecular features (random).
        
        Args:
            n_molecules: Number of low-symmetry molecules
            n_atoms: Number of atoms per molecule
            n_features: Number of orbital features per atom
            random_seed: Random seed for reproducibility
            
        Returns:
            Features array of shape (n_molecules, n_atoms, n_features)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random features without symmetry constraints
        features = np.random.normal(
            loc=self.qm9_stats['mean'],
            scale=self.qm9_stats['std'] * 0.4,
            size=(n_molecules, n_atoms, n_features)
        )
        
        # Add noise
        features += np.random.normal(
            loc=0,
            scale=self.qm9_stats['std'] * 0.25,
            size=(n_molecules, n_atoms, n_features)
        )
        
        return features
    
    def generate_complete_dataset(self, n_total: int = 2000, high_symmetry_ratio: float = 0.25,
                                 n_atoms: int = 9, n_features: int = 5, 
                                 random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete dataset with high and low symmetry molecules.
        
        Args:
            n_total: Total number of molecules
            high_symmetry_ratio: Proportion of high-symmetry molecules
            n_atoms: Number of atoms per molecule
            n_features: Number of orbital features per atom
            random_seed: Random seed for reproducibility
            
        Returns:
            (features, labels) where labels: 1 = high symmetry, 0 = low symmetry
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n_high = int(n_total * high_symmetry_ratio)
        n_low = n_total - n_high
        
        print(f"\nGenerating dataset:")
        print(f"  Total molecules: {n_total}")
        print(f"  High-symmetry: {n_high} ({high_symmetry_ratio*100:.1f}%)")
        print(f"  Low-symmetry: {n_low} ({(1-high_symmetry_ratio)*100:.1f}%)")
        
        # Generate high-symmetry features
        high_features = self.generate_high_symmetry_features(
            n_high, n_atoms, n_features, random_seed
        )
        high_labels = np.ones(n_high, dtype=int)
        
        # Generate low-symmetry features
        low_features = self.generate_low_symmetry_features(
            n_low, n_atoms, n_features, random_seed + 1
        )
        low_labels = np.zeros(n_low, dtype=int)
        
        # Combine and shuffle
        features = np.vstack([high_features, low_features])
        labels = np.concatenate([high_labels, low_labels])
        
        # Shuffle dataset
        indices = np.random.permutation(n_total)
        features = features[indices]
        labels = labels[indices]
        
        print(f"\nDataset generated successfully!")
        print(f"  Feature shape: {features.shape}")
        print(f"  Label distribution: {np.bincount(labels)}")
        
        return features, labels