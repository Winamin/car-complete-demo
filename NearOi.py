#!/usr/bin/env python3

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Any, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from enum import Enum
import random

@dataclass
class Neuron:
    """
    神经元：具有量化意识的基本单元
    论文公式 (1): C_i 的完整实现
    """
    layer: int
    index: int
    B: float = 0.5      # 基础意识强度
    r: float = 0.0      # 响应率
    v: float = 0.5      # 激活水平
    C: float = 0.0      # 最终意识强度
    
    def compute_f(self) -> float:
        """社交信号 f_j = tanh(C_j) - 论文定义"""
        return math.tanh(self.C)
    
    def update_state(self, accuracy: float, learning_rate: float):
        """论文学习规则：v_i ← 0.9·v_i + 0.1·accuracy"""
        self.v = 0.9 * self.v + 0.1 * accuracy
        self.v = np.clip(self.v, 0.0, 1.0)

@dataclass
class KnowledgeTriple:
    """
    知识三元组: (Condition, Action, Confidence)
    论文 Definition: Knowledge Triple
    """
    condition: Dict[str, Any]
    action: Dict[str, Any] 
    confidence: Dict[str, float]
    creator_neuron: Tuple[int, int] = None
    use_count: int = 0
    last_used_step: int = 0
    evolution_history: List = field(default_factory=list)

@dataclass
class MathematicalConcept:
    """
    数学概念：基于Rosch原型理论
    match(x) = 𝕀[cos(x, prototype_c) > 1 - boundary_c]
    
    关键改进：只包含纯数学概念，无物理预设
    """
    name: str
    prototype: np.ndarray
    boundary: float = 0.3
    abstract_level: int = 0
    definition: str = ""
    
    def match(self, x: np.ndarray) -> bool:
        """概念匹配函数"""
        if len(x) != len(self.prototype):
            return False
        
        cos_sim = np.dot(x, self.prototype) / (
            np.linalg.norm(x) * np.linalg.norm(self.prototype) + 1e-10
        )
        
        return cos_sim > (1 - self.boundary)

@dataclass
class MathematicalPattern:
    """
    发现的数学模式 - 零样本自发明
    """
    pattern_id: str
    pattern_type: str
    mathematical_signature: str
    confidence: float
    supporting_evidence: List[float]
    self_invented_name: str = ""
    first_principles_derivation: str = ""

class TrueZeroNearOi:
    """
    真正的零样本NearOi实现
    
    三层架构（完全重新设计）：
    1. Neural Layer: 意识强度量化（论文公式1）
    2. Mathematical Layer: 纯数学模式检测
    3. Conceptual Layer: 自发明概念系统
    """
    
    def __init__(self, layers: int = 5, neurons_per_layer: int = 2000):
        """
        初始化系统 - 论文 Algorithm 1: NearOi Initialization
        """
        # 论文参数
        self.epsilon = 0.01
        self.alpha = 0.3
        self.eta = 0.5
        self.w_max = 5.0
        self.lambda_lr = 0.05
        
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.E_max = layers * neurons_per_layer
        
        # 核心组件
        self.neurons: List[List[Neuron]] = []
        self.knowledge_base: List[KnowledgeTriple] = []
        self.mathematical_concepts: Dict[str, MathematicalConcept] = {}
        self.discovered_patterns: List[MathematicalPattern] = []
        
        # 系统状态
        self.trust_weights = defaultdict(lambda: 1.0)
        self.step_count = 0
        self.reasoning_trace = []
        self.invented_terminology = {}  # 自发明的科学语言
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """论文 Algorithm 1: NearOi Initialization"""
        # 1. 初始化神经网络
        for layer_idx in range(self.layers):
            layer = []
            for neuron_idx in range(self.neurons_per_layer):
                neuron = Neuron(
                    layer=layer_idx,
                    index=neuron_idx,
                    B=np.random.uniform(0.3, 0.6),  # 论文：随机初始化B
                    r=0.0,
                    v=0.5
                )
                layer.append(neuron)
            self.neurons.append(layer)
        
        # 2. 初始化数学概念层（零预设）
        self._init_mathematical_concepts()
        
        # 3. 初始化基础知识库（零预设）
        self._init_zero_knowledge_base()
    
    def _init_mathematical_concepts(self):
        """初始化纯数学概念（零物理预设）"""
        # 只有基础数学概念，无任何物理术语
        
        self.mathematical_concepts['linear'] = MathematicalConcept(
            name='linear',
            prototype=np.array([1.0, 0.0, 0.0, 0.0]),
            boundary=0.4,
            abstract_level=1,
            definition='Linear mathematical relationship'
        )
        
        self.mathematical_concepts['periodic'] = MathematicalConcept(
            name='periodic',
            prototype=np.array([0.0, 1.0, 0.0, 0.0]),
            boundary=0.35,
            abstract_level=1,
            definition='Recurring mathematical pattern'
        )
        
        self.mathematical_concepts['symmetric'] = MathematicalConcept(
            name='symmetric',
            prototype=np.array([0.0, 0.0, 1.0, 0.0]),
            boundary=0.4,
            abstract_level=2,
            definition='Mathematical invariance under transformation'
        )
        
        self.mathematical_concepts['complex'] = MathematicalConcept(
            name='complex',
            prototype=np.array([0.0, 0.0, 0.0, 1.0]),
            boundary=0.45,
            abstract_level=2,
            definition='High-dimensional mathematical structure'
        )
    
    def _init_zero_knowledge_base(self):
        """初始化零知识库（只有基础逻辑）"""
        # 只有最基础的形式逻辑，无物理知识
        
        self.knowledge_base.append(KnowledgeTriple(
            condition={
                'pattern_type': 'unknown',
                'context': 'mathematical_analysis',
                'constraints': []
            },
            action={
                'operation': 'mathematical_exploration',
                'parameters': {'method': 'pattern_detection'},
                'expected_outcome': 'mathematical_structure'
            },
            confidence={
                'belief': 0.5,  # 初始低置信度
                'support': 0.3,
                'success_rate': 0.0,
                'last_used': 0
            }
        ))
    
    def compute_consciousness_intensity(
            self,
            neuron: Neuron,
            active_neurons: List[Neuron]
    ) -> float:
        """
        论文公式 (1): C_i 的完整计算
        实现论文中的完整意识强度计算
        """
        ℓ_i = neuron.layer
        i = neuron.index

        if len(active_neurons) == 0:
            # 无其他神经元时的自激活
            self_activation = neuron.B + self.alpha * neuron.r * neuron.v
            return np.clip(self_activation, 0.0, 1.0)

        # 计算注意力权重
        v_vals = [n.v for n in active_neurons]
        v_mean = np.mean(v_vals)
        noise = np.sqrt(np.mean([(v - v_mean)**2 for v in v_vals]))
        
        prod = np.mean([n.r * n.v for n in active_neurons])
        delta = self.lambda_lr * noise * prod

        layer_neurons = [n for n in active_neurons if n.layer == ℓ_i]
        denominator = sum(n.r * n.v for n in layer_neurons) + delta
        
        if denominator < 1e-10:
            attention = self.epsilon
        else:
            attention = (neuron.r * neuron.v) / denominator

        # 计算社交影响
        social_influence = 0.0
        for other in active_neurons:
            if other.layer == ℓ_i and other.index == i:
                continue

            w_ij = min(
                self.trust_weights[(neuron.layer, neuron.index, other.layer, other.index)],
                self.w_max
            )

            ℓ_j = other.layer
            layer_decay = math.exp(-self.eta * abs(ℓ_i - ℓ_j))
            f_j = other.compute_f()

            social_influence += w_ij * layer_decay * f_j

        # 论文公式 (1): C_i = attention * (B_i + α*r_i*v_i + social_influence)
        C_i = attention * (
            neuron.B +
            self.alpha * neuron.r * neuron.v +
            social_influence
        )

        return np.clip(C_i, 0.0, 1.0)
    
    def generate_mathematical_signature(self, data: np.ndarray) -> Dict[str, float]:
        """
        生成数据的纯数学特征签名
        无任何物理预设，完全基于数学分析
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        signature = {}
        
        # 基础统计特征
        signature['mean'] = float(np.mean(data))
        signature['variance'] = float(np.var(data))
        signature['skewness'] = self._compute_skewness(data)
        signature['kurtosis'] = self._compute_kurtosis(data)
        
        # 自相关特征
        if data.shape[0] > 10:
            signature['autocorr_lag1'] = self._compute_autocorrelation(data[:, 0], lag=1)
            signature['autocorr_lag2'] = self._compute_autocorrelation(data[:, 0], lag=2)
        
        # 频域特征
        if data.shape[0] > 20:
            freq_features = self._compute_frequency_features(data[:, 0])
            signature.update(freq_features)
        
        # 熵和复杂度
        signature['entropy'] = self._compute_entropy(data[:, 0])
        signature['complexity'] = self._compute_complexity(data[:, 0])
        
        # 几何特征
        if data.shape[1] > 1:
            signature['correlation'] = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
            if not np.isnan(signature['correlation']):
                signature['geometric_structure'] = self._analyze_geometric_structure(data)
        
        return signature
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        data_flat = data.flatten()
        mean = np.mean(data_flat)
        std = np.std(data_flat)
        if std == 0:
            return 0.0
        return float(np.mean(((data_flat - mean) / std) ** 3))
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        data_flat = data.flatten()
        mean = np.mean(data_flat)
        std = np.std(data_flat)
        if std == 0:
            return 0.0
        return float(np.mean(((data_flat - mean) / std) ** 4))
    
    def _compute_autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """计算自相关"""
        if len(data) <= lag:
            return 0.0
        
        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return 0.0
        
        autocorr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
        return float(autocorr) if not np.isnan(autocorr) else 0.0
    
    def _compute_frequency_features(self, data: np.ndarray) -> Dict[str, float]:
        """计算频域特征"""
        fft = np.fft.fft(data)
        power_spectrum = np.abs(fft) ** 2
        
        freqs = np.fft.fftfreq(len(data))
        peak_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        peak_freq = abs(freqs[peak_freq_idx])
        
        power_normalized = power_spectrum / np.sum(power_spectrum)
        entropy = -np.sum(power_normalized * np.log2(power_normalized + 1e-10))
        
        return {
            'peak_frequency': float(peak_freq),
            'frequency_entropy': float(entropy),
            'spectral_centroid': float(np.sum(np.abs(freqs) * power_spectrum) / np.sum(power_spectrum))
        }
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """计算信息熵"""
        data_discrete = np.histogram(data, bins=20)[0]
        data_normalized = data_discrete / np.sum(data_discrete)
        
        entropy = -np.sum(data_normalized * np.log2(data_normalized + 1e-10))
        return float(entropy)
    
    def _compute_complexity(self, data: np.ndarray) -> float:
        """计算时间序列复杂度"""
        if len(data) < 4:
            return 0.0
        
        m = 2
        r = 0.2 * np.std(data)
        
        phi_m = self._approximate_entropy(data, m, r)
        phi_m1 = self._approximate_entropy(data, m + 1, r)
        
        complexity = phi_m - phi_m1
        return float(complexity)
    
    def _approximate_entropy(self, data: np.ndarray, m: int, r: float) -> float:
        """计算近似熵"""
        def _maxdist(xi: np.ndarray, xj: np.ndarray, m: int) -> float:
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        N = len(data)
        patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
        
        C = 0.0
        for i in range(N - m + 1):
            template = patterns[i]
            matches = [_maxdist(template, patterns[j], m) <= r for j in range(N - m + 1)]
            C += sum(matches) / (N - m + 1)
        
        phi = C / (N - m + 1)
        return math.log(phi + 1e-10) if phi > 0 else -float('inf')
    
    def _analyze_geometric_structure(self, data: np.ndarray) -> float:
        """分析几何结构"""
        if data.shape[1] < 2:
            return 0.0
        
        distances = np.linalg.norm(data, axis=1)
        
        if len(distances) > 0:
            cv = np.std(distances) / (np.mean(distances) + 1e-10)
            structure_score = 1.0 / (1.0 + cv)
            return float(structure_score)
        
        return 0.0
    
    def discover_mathematical_patterns(self, signature: Dict[str, float]) -> List[MathematicalPattern]:
        """
        从数学签名中发现模式 - 真正的零样本发现
        """
        patterns = []
        
        # 1. 周期性模式检测
        if signature.get('peak_frequency', 0) > 0.01:
            pattern = self._create_periodic_pattern(signature)
            patterns.append(pattern)
        
        # 2. 相关性模式检测
        if abs(signature.get('correlation', 0)) > 0.5:
            pattern = self._create_correlation_pattern(signature)
            patterns.append(pattern)
        
        # 3. 对称性模式检测
        if signature.get('geometric_structure', 0) > 0.3:
            pattern = self._create_symmetry_pattern(signature)
            patterns.append(pattern)
        
        # 4. 自相关模式检测
        if abs(signature.get('autocorr_lag1', 0)) > 0.3:
            pattern = self._create_autocorr_pattern(signature)
            patterns.append(pattern)
        
        # 5. 复杂模式检测
        if signature.get('complexity', 0) > 0.1:
            pattern = self._create_complexity_pattern(signature)
            patterns.append(pattern)
        
        return patterns
    
    def _create_periodic_pattern(self, signature: Dict[str, float]) -> MathematicalPattern:
        """创建周期性模式发现"""
        pattern_id = f"PERIODIC_{len(self.discovered_patterns):03d}"
        
        peak_freq = signature.get('peak_frequency', 0)
        freq_entropy = signature.get('frequency_entropy', 0)
        
        # 自发明名称
        name = self._generate_self_invented_name("periodic", [f"freq_{peak_freq:.3f}"])
        
        return MathematicalPattern(
            pattern_id=pattern_id,
            pattern_type="periodic",
            mathematical_signature=f"sin(2πft) with f={peak_freq:.3f}",
            confidence=min(0.9, 0.5 + freq_entropy / 10),
            supporting_evidence=[peak_freq, freq_entropy],
            self_invented_name=name,
            first_principles_derivation="Periodicity emerges from frequency domain analysis showing dominant frequency components"
        )
    
    def _create_correlation_pattern(self, signature: Dict[str, float]) -> MathematicalPattern:
        """创建相关性模式发现"""
        pattern_id = f"CORRELATION_{len(self.discovered_patterns):03d}"
        
        correlation = signature.get('correlation', 0)
        variance = signature.get('variance', 0)
        
        name = self._generate_self_invented_name("correlated", [f"corr_{correlation:.3f}"])
        
        return MathematicalPattern(
            pattern_id=pattern_id,
            pattern_type="correlated",
            mathematical_signature=f"linear_dependency(r={correlation:.3f})",
            confidence=min(0.9, abs(correlation)),
            supporting_evidence=[abs(correlation), variance],
            self_invented_name=name,
            first_principles_derivation="Correlation arises from statistical dependence analysis between data dimensions"
        )
    
    def _create_symmetry_pattern(self, signature: Dict[str, float]) -> MathematicalPattern:
        """创建对称性模式发现"""
        pattern_id = f"SYMMETRY_{len(self.discovered_patterns):03d}"
        
        structure = signature.get('geometric_structure', 0)
        complexity = signature.get('complexity', 0)
        
        name = self._generate_self_invented_name("symmetric", [f"struct_{structure:.3f}"])
        
        return MathematicalPattern(
            pattern_id=pattern_id,
            pattern_type="symmetric",
            mathematical_signature=f"invariant_under_transformation({structure:.3f})",
            confidence=structure,
            supporting_evidence=[structure, complexity],
            self_invented_name=name,
            first_principles_derivation="Symmetry detected through geometric structure analysis showing reduced variance in spatial patterns"
        )
    
    def _create_autocorr_pattern(self, signature: Dict[str, float]) -> MathematicalPattern:
        """创建自相关模式发现"""
        pattern_id = f"AUTOCORR_{len(self.discovered_patterns):03d}"
        
        autocorr1 = signature.get('autocorr_lag1', 0)
        autocorr2 = signature.get('autocorr_lag2', 0)
        
        name = self._generate_self_invented_name("autocorrelated", [f"mem_{autocorr1:.3f}"])
        
        return MathematicalPattern(
            pattern_id=pattern_id,
            pattern_type="autocorrelated",
            mathematical_signature=f"memory_effect(r1={autocorr1:.3f}, r2={autocorr2:.3f})",
            confidence=max(abs(autocorr1), abs(autocorr2)),
            supporting_evidence=[abs(autocorr1), abs(autocorr2)],
            self_invented_name=name,
            first_principles_derivation="Temporal memory detected through autocorrelation analysis showing future dependence on past values"
        )
    
    def _create_complexity_pattern(self, signature: Dict[str, float]) -> MathematicalPattern:
        """创建复杂度模式发现"""
        pattern_id = f"COMPLEX_{len(self.discovered_patterns):03d}"
        
        complexity = signature.get('complexity', 0)
        entropy = signature.get('entropy', 0)
        
        name = self._generate_self_invented_name("complex", [f"compl_{complexity:.3f}"])
        
        return MathematicalPattern(
            pattern_id=pattern_id,
            pattern_type="complex",
            mathematical_signature=f"irregular_pattern(C={complexity:.3f}, H={entropy:.3f})",
            confidence=min(0.8, complexity),
            supporting_evidence=[complexity, entropy],
            self_invented_name=name,
            first_principles_derivation="Complexity emerges from approximate entropy analysis showing pattern irregularity and unpredictability"
        )
    
    def _generate_self_invented_name(self, pattern_type: str, characteristics: List[str]) -> str:
        """自发明科学术语"""
        # 基于数学特性生成独特名称
        type_prefixes = {
            'periodic': 'CYCLO',
            'correlated': 'LINK',
            'symmetric': 'MIRR',
            'autocorrelated': 'MEM',
            'complex': 'LABY'
        }
        
        prefix = type_prefixes.get(pattern_type, 'MATH')
        index = f"{len(self.discovered_patterns):03d}"
        char_code = ''.join([c.replace('.', '_') for c in characteristics[:2]])
        
        generated_name = f"{prefix}_{char_code}_{index}"
        
        # 记录术语定义
        self.invented_terminology[generated_name] = {
            'pattern_type': pattern_type,
            'characteristics': characteristics,
            'first_discovery_time': time.time(),
            'mathematical_basis': f"Discovered from {pattern_type} analysis"
        }
        
        return generated_name
    
    def activate_mathematical_concepts(self, features: np.ndarray) -> List[str]:
        """
        概念层激活
        match(x) = 𝕀[cos(x, prototype_c) > 1 - boundary_c]
        """
        activated = []
        
        for name, concept in self.mathematical_concepts.items():
            # 添加随机边界扰动（论文中的随机性）
            random_boundary = concept.boundary + np.random.uniform(-0.05, 0.05)
            random_boundary = np.clip(random_boundary, 0.1, 0.8)
            
            original_boundary = concept.boundary
            concept.boundary = random_boundary
            
            if concept.match(features):
                activated.append(name)
            
            concept.boundary = original_boundary
        
        return activated
    
    def neural_layer_inference(self, task: Dict[str, Any]) -> List[Neuron]:
        """
        神经层推理 - 论文8阶段推理管道
        """
        active_neurons = []
        
        # 计算所有神经元的意识强度
        for layer in self.neurons:
            for neuron in layer:
                neuron.C = self.compute_consciousness_intensity(neuron, active_neurons)
                
                if neuron.C > 0.3:  # 激活阈值
                    active_neurons.append(neuron)
                    neuron.r = (neuron.r * (self.step_count - 1) + 1) / self.step_count
        
        # 选择前5个最高意识强度的神经元
        top_neurons = sorted(active_neurons, key=lambda n: n.C, reverse=True)[:5]
        
        return top_neurons
    
    def zero_shot_theory_construction(self, patterns: List[MathematicalPattern]) -> Dict:
        """
        零样本理论构建 - 从数学模式到理论
        """
        if not patterns:
            return {
                'theory_type': 'no_pattern_detected',
                'expression': 'F(x) = random',
                'confidence': 0.1,
                'derivation_steps': ['No mathematical patterns detected']
            }
        
        # 基于发现的模式构建理论
        pattern_types = [p.pattern_type for p in patterns]
        unique_types = list(set(pattern_types))
        
        theory_name = self._generate_theory_name(patterns)
        mathematical_expression = self._construct_mathematical_expression(patterns)
        derivation_steps = self._generate_derivation_steps(patterns)
        
        theory = {
            'theory_type': f"{'_'.join(unique_types)}_theory",
            'theory_name': theory_name,
            'expression': mathematical_expression,
            'confidence': np.mean([p.confidence for p in patterns]),
            'derivation_steps': derivation_steps,
            'patterns_found': len(patterns),
            'self_invented_terms': len(self.invented_terminology),
            'mathematical_novelty': 'Discovered from pure mathematical analysis'
        }
        
        return theory
    
    def _generate_theory_name(self, patterns: List[MathematicalPattern]) -> str:
        """生成理论名称"""
        if not patterns:
            return "UNKNOWN_THEORY"
        
        pattern_types = [p.pattern_type for p in patterns]
        unique_types = list(set(pattern_types))
        
        if len(unique_types) == 1:
            return f"{unique_types[0].upper()}_THEORY_V{len(self.discovered_patterns)}"
        elif len(unique_types) == 2:
            return f"{unique_types[0].upper()}_{unique_types[1].upper()}_THEORY_V{len(self.discovered_patterns)}"
        else:
            return f"MULTI_PATTERN_THEORY_V{len(self.discovered_patterns)}"
    
    def _construct_mathematical_expression(self, patterns: List[MathematicalPattern]) -> str:
        """构造数学表达式"""
        expressions = [p.mathematical_signature for p in patterns]
        return f"Mathematical framework: {' ∪ '.join(expressions)}"
    
    def _generate_derivation_steps(self, patterns: List[MathematicalPattern]) -> List[str]:
        """生成推导步骤"""
        steps = ["Zero-shot theory construction from mathematical patterns:"]
        
        for i, pattern in enumerate(patterns, 1):
            steps.append(f"{i}. {pattern.self_invented_name}: {pattern.first_principles_derivation}")
        
        return steps
    
    def zero_sample_scientific_discovery(self, raw_data: Dict[str, np.ndarray]) -> Dict:
        """
        零样本科学发现 - 论文核心功能
        完全从零开始，无任何预设知识
        """
        print("🧠 TrueZeroNearOi: 开始零样本科学发现...")
        print("📊 处理数据维度:", {k: v.shape for k, v in raw_data.items()})
        
        self.step_count += 1
        self.reasoning_trace = []
        
        all_patterns = []
        all_signatures = {}
        
        # 第一阶段：纯数学签名生成
        for data_name, data_array in raw_data.items():
            signature = self.generate_mathematical_signature(data_array)
            all_signatures[data_name] = signature
            
            # 第二阶段：数学模式发现
            patterns = self.discover_mathematical_patterns(signature)
            all_patterns.extend(patterns)
        
        # 激活数学概念
        if all_signatures:
            first_signature = list(all_signatures.values())[0]
            features = np.array(list(first_signature.values())[:4])  # 取前4个特征
            activated_concepts = self.activate_mathematical_concepts(features)
        else:
            activated_concepts = []
        
        # 神经层推理
        top_neurons = self.neural_layer_inference({})
        
        # 组合发现
        combined_patterns = self._combine_patterns(all_patterns)
        
        # 零样本理论构建
        theory = self.zero_shot_theory_construction(combined_patterns)
        
        # 学习更新
        self._learning_update(theory, top_neurons)
        
        return {
            'mathematical_signatures': all_signatures,
            'discovered_patterns': [p.__dict__ for p in combined_patterns],
            'activated_concepts': activated_concepts,
            'neural_contributors': len(top_neurons),
            'theory': theory,
            'self_invented_terminology': self.invented_terminology,
            'overall_confidence': theory['confidence'],
            'reasoning_steps': len(combined_patterns),
            'step_count': self.step_count
        }
    
    def _combine_patterns(self, patterns: List[MathematicalPattern]) -> List[MathematicalPattern]:
        """组合发现的模式"""
        combined = patterns.copy()
        
        if len(patterns) > 1:
            # 创建组合模式
            combined_pattern = MathematicalPattern(
                pattern_id=f"COMBINED_{len(self.discovered_patterns):03d}",
                pattern_type="combined",
                mathematical_signature=f"Σ({', '.join(set([p.pattern_type for p in patterns]))})",
                confidence=np.mean([p.confidence for p in patterns]),
                supporting_evidence=[p.confidence for p in patterns],
                self_invented_name=self._generate_self_invented_name("combined", ["multi"]),
                first_principles_derivation="Combined pattern emerges when multiple independent mathematical signatures are simultaneously present"
            )
            combined.append(combined_pattern)
        
        # 更新发现的模式列表
        self.discovered_patterns.extend(combined)
        
        return combined
    
    def _learning_update(self, theory: Dict, top_neurons: List[Neuron]):
        """
        学习更新 - 论文中的关键更新规则
        B_i ← B_i + λ(accuracy - C_i)
        v_i ← 0.9·v_i + 0.1·accuracy
        """
        accuracy = theory['confidence']
        
        for neuron in top_neurons:
            # 论文更新规则
            neuron.B += self.lambda_lr * (accuracy - neuron.C)
            neuron.B = np.clip(neuron.B, 0.0, 1.0)
            
            neuron.update_state(accuracy, self.lambda_lr)
        
        # 信任权重更新
        if len(top_neurons) >= 2 and accuracy > 0.8:
            for i in range(len(top_neurons) - 1):
                n1, n2 = top_neurons[i], top_neurons[i + 1]
                key = (n1.layer, n1.index, n2.layer, n2.index)
                self.trust_weights[key] = min(
                    self.trust_weights[key] + 0.1,
                    self.w_max
                )

# 零样本科学发现测试
def create_neutral_scientific_data():
    """创建真正中性的科学数据"""
    print("🔬 创建中性科学数据...")
    
    t = np.linspace(0, 4*np.pi, 100)
    
    # 数据1: 纯周期性（无物理暗示）
    periodic = np.sin(2*np.pi*0.1*t) + 0.2*np.sin(2*np.pi*0.3*t)
    
    # 数据2: 相关性数据
    x = np.sin(t)
    y = 0.7*np.sin(t + np.pi/4) + 0.1*np.random.randn(len(t))
    correlated = np.column_stack([x, y])
    
    # 数据3: 几何结构
    radius = 2 + 0.5*np.sin(3*t)
    angles = t
    geometric = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    
    return {
        'periodic_signal': periodic,
        'correlated_components': correlated,
        'geometric_structure': geometric,
        'baseline_noise': np.random.randn(100) * 0.05
    }

def test_zero_sample_discovery():
    """测试零样本科学发现"""
    print("🚀 TrueZeroNearOi - 零样本科学发现测试")
    print("=" * 60)
    
    # 创建系统
    system = TrueZeroNearOi(layers=5, neurons_per_layer=100)
    
    # 创建中性数据
    data = create_neutral_scientific_data()
    
    # 执行零样本发现
    start_time = time.time()
    result = system.zero_sample_scientific_discovery(data)
    discovery_time = time.time() - start_time
    
    print(f"\n🎯 零样本发现结果:")
    print(f"⏱️ 执行时间: {discovery_time:.4f}秒")
    print(f"🧠 激活概念: {len(result['activated_concepts'])}")
    print(f"🔬 发现模式: {len(result['discovered_patterns'])}")
    print(f"🏷️ 自发明术语: {len(result['self_invented_terminology'])}")
    print(f"📊 整体置信度: {result['overall_confidence']:.3f}")
    
    print(f"\n📋 理论构建:")
    theory = result['theory']
    print(f"理论名称: {theory['theory_name']}")
    print(f"数学表达式: {theory['expression']}")
    print(f"推导步骤: {len(theory['derivation_steps'])}步")
    
    print(f"\n🔍 发现的模式:")
    for i, pattern in enumerate(result['discovered_patterns'][:5], 1):
        print(f"{i}. {pattern['self_invented_name']}: {pattern['mathematical_signature']}")
    
    print(f"\n🏷️ 自发明术语库:")
    for term, info in list(result['self_invented_terminology'].items())[:5]:
        print(f"  {term}: {info['mathematical_basis']}")
    
    return result

if __name__ == "__main__":
    test_result = test_zero_sample_discovery()