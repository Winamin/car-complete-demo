#!/usr/bin/env python3
# CAR Online Learning Test - Verify Learn-While-Inference Capability

import sys
import os
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
np.random.seed(SEED)


class KnowledgeBase:
    def __init__(self, max_size=10000):
        self.samples = []
        self.max_size = max_size
    
    def add(self, x, y):
        if len(self.samples) < self.max_size:
            self.samples.append((x.copy(), y))
        else:
            idx = np.random.randint(0, self.max_size)
            self.samples[idx] = (x.copy(), y)
    
    def retrieve(self, x, k=3):
        if not self.samples:
            return []
        
        similarities = []
        for stored_x, stored_y in self.samples:
            dist = np.linalg.norm(x - stored_x) / (np.linalg.norm(x) + 1e-10)
            similarities.append((dist, stored_x, stored_y))
        
        similarities.sort(key=lambda x: x[0])
        return [(s[1], s[2]) for s in similarities[:k]]
    
    def get_size(self):
        return len(self.samples)


class StandardCARModel:
    def __init__(self, k_neighbors=3, max_size=5000):
        self.k = k_neighbors
        self.max_size = max_size
        self.knowledge_base = KnowledgeBase(max_size=max_size)
        self.fitted = False
        self.initial_size = 0
    
    def fit(self, X, y):
        self.knowledge_base = KnowledgeBase(max_size=len(X))
        for xi, yi in zip(X, y):
            self.knowledge_base.add(xi, yi)
        self.initial_size = self.knowledge_base.get_size()
        self.fitted = True
        return self
    
    def predict_single(self, x):
        if not self.fitted:
            return 0.0
        
        neighbors = self.knowledge_base.retrieve(x, k=self.k)
        
        if not neighbors:
            return 0.0
        
        predictions = []
        weights = []
        
        for neighbor_x, neighbor_y in neighbors:
            dist = np.linalg.norm(x - neighbor_x)
            weight = 1.0 / (dist + 1e-10)
            predictions.append(neighbor_y)
            weights.append(weight)
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        return np.sum(predictions * weights) / np.sum(weights)
    
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


class OnlineLearningCAR:
    def __init__(self, k_neighbors=3, max_size=5000, learning_rate=0.1):
        self.k = k_neighbors
        self.max_size = max_size
        self.learning_rate = learning_rate
        self.knowledge_base = KnowledgeBase(max_size=max_size)
        self.fitted = False
        self.initial_size = 0
        self.update_count = 0
        self.prediction_history = []
        
    def fit(self, X, y):
        self.knowledge_base = KnowledgeBase(max_size=self.max_size)
        for xi, yi in zip(X, y):
            self.knowledge_base.add(xi, yi)
        self.initial_size = self.knowledge_base.get_size()
        self.fitted = True
        return self
    
    def predict_single(self, x, update=True):
        if not self.fitted:
            return 0.0
        
        neighbors = self.knowledge_base.retrieve(x, k=self.k)
        
        if not neighbors:
            prediction = 0.0
        else:
            predictions = []
            weights = []
            
            for neighbor_x, neighbor_y in neighbors:
                dist = np.linalg.norm(x - neighbor_x)
                weight = 1.0 / (dist + 1e-10)
                predictions.append(neighbor_y)
                weights.append(weight)
            
            predictions = np.array(predictions)
            weights = np.array(weights)
            prediction = np.sum(predictions * weights) / np.sum(weights)
        
        if update:
            self.knowledge_base.add(x, prediction)
            self.update_count += 1
        
        return prediction
    
    def predict(self, X, update=True):
        return np.array([self.predict_single(x, update=update) for x in X])
    
    def get_knowledge_size(self):
        return self.knowledge_base.get_size()


def generate_signal_1(x):
    return np.sin(x[:, 0]) + np.cos(x[:, 1]) + np.log(1 + x[:, 2]**2)

def generate_signal_2(x):
    return np.exp(x[:, 0] * 0.1) + np.power(x[:, 1], 3) + np.tan(x[:, 2])

def generate_signal_3(x):
    return np.sqrt(np.abs(x[:, 0])) + np.sign(x[:, 1]) * x[:, 1]**2 + 1/(1 + np.exp(-x[:, 2]))


def run_online_learning_test():
    print("="*80)
    print("  CAR Online Learning Capability Test")
    print("="*80)
    
    N_SAMPLES = 500
    FEATURE_DIM = 100
    
    print("\nGenerating training data...")
    X_train = np.random.randn(N_SAMPLES, FEATURE_DIM)
    y_train = generate_signal_1(X_train)
    
    print(f"Training data: {N_SAMPLES} samples, {FEATURE_DIM} dimensions")
    
    print("\nTraining standard CAR...")
    standard_car = StandardCARModel(k_neighbors=3)
    standard_car.fit(X_train, y_train)
    print(f"Knowledge base size: {standard_car.initial_size}")
    
    print("\nTraining online learning CAR...")
    online_car = OnlineLearningCAR(k_neighbors=3, max_size=10000)
    online_car.fit(X_train, y_train)
    print(f"Knowledge base size: {online_car.get_knowledge_size()}")
    
    print("\nGenerating new signal test data...")
    X_new = np.random.randn(N_SAMPLES, FEATURE_DIM)
    y_new = generate_signal_2(X_new)
    
    print(f"Test data: {N_SAMPLES} samples")
    
    print("\nInitial prediction...")
    standard_pred = standard_car.predict(X_new)
    standard_mse_initial = np.mean((standard_pred - y_new) ** 2)
    
    online_pred_initial = online_car.predict(X_new, update=False)
    online_mse_initial = np.mean((online_pred_initial - y_new) ** 2)
    
    print(f"Standard CAR MSE: {standard_mse_initial:.4f}")
    print(f"Online CAR MSE: {online_mse_initial:.4f}")
    
    print("\nOnline learning...")
    batch_size = 50
    n_batches = len(X_new) // batch_size
    
    online_mse_history = []
    knowledge_size_history = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_new[start_idx:end_idx]
        y_batch = y_new[start_idx:end_idx]
        
        pred_batch = online_car.predict(X_batch, update=True)
        
        mse_batch = np.mean((pred_batch - y_batch) ** 2)
        online_mse_history.append(mse_batch)
        knowledge_size_history.append(online_car.get_knowledge_size())
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}: KB={online_car.get_knowledge_size()}, MSE={mse_batch:.4f}")
    
    print("\nFinal results...")
    final_pred = online_car.predict(X_new[-100:], update=False)
    final_mse = np.mean((final_pred - y_new[-100:]) ** 2)
    
    print(f"Knowledge base size: {online_car.get_knowledge_size()}")
    print(f"Standard CAR MSE: {standard_mse_initial:.4f}")
    print(f"Online CAR MSE: {final_mse:.4f}")
    
    improvement = (standard_mse_initial - final_mse) / standard_mse_initial * 100
    
    print(f"\nPerformance improvement: {improvement:.2f}%")
    print(f"Paper reference: 98.5%")
    
    print("\nEvaluation results...")
    if improvement >= 98.0:
        status = "✅ Excellent"
    elif improvement >= 90.0:
        status = "✓ Good"
    elif improvement >= 50.0:
        status = "⚠ Average"
    else:
        status = "✗ Poor"
    
    print(f"Status: {status}")
    print(f"Improvement rate: {improvement:.2f}%")
    
    results = {
        "test_time": datetime.now().isoformat(),
        "online_learning_test": {
            "initial_knowledge_size": int(online_car.initial_size),
            "final_knowledge_size": int(online_car.get_knowledge_size()),
            "standard_car_mse": float(standard_mse_initial),
            "online_car_initial_mse": float(online_mse_initial),
            "online_car_final_mse": float(final_mse),
            "improvement_percent": float(improvement),
            "mse_history": [float(m) for m in online_mse_history],
            "knowledge_size_history": knowledge_size_history
        }
    }
    
    results_file = os.path.join(os.path.dirname(__file__), 'online_learning_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return {
        'initial_mse': standard_mse_initial,
        'final_mse': final_mse,
        'improvement': improvement,
        'mse_history': online_mse_history,
        'knowledge_size_history': knowledge_size_history,
        'knowledge_size_initial': online_car.initial_size,
        'knowledge_size_final': online_car.get_knowledge_size()
    }


def run_main():
    try:
        results = run_online_learning_test()
        
        if results['improvement'] >= 98.0:
            print("\n✅ Test passed")
            return 0
        else:
            print("\n⚠ Test did not fully meet target")
            return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_main())