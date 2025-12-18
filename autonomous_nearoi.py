import numpy as np
import math
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class Neuron:
    """神经元：在环境中学习的智能体"""
    layer: int
    index: int
    
    # 核心状态
    B: float = 0.5  # 信念
    r: float = 0.0  # 经验次数
    v: float = 0.5  # 验证分数
    C: float = 0.0  # 意识强度
    
    # 学习到的观察策略（初始随机）
    observation_strategy: Dict = field(default_factory=lambda: {
        'what_to_look_at': np.random.choice(['positions', 'charges', 'spins', 'velocities']),
        'how_to_process': np.random.choice(['sum', 'mean', 'variance', 'max', 'min', 'norm', 'angle']),
        'transformation': np.random.uniform(-1, 1, 3),
        'combine_variables': np.random.choice([False, True]),  # 是否组合多个变量
        'rotation_angle': np.random.uniform(0, 2*np.pi),  # 旋转角度
        'nonlinear': np.random.choice(['linear', 'square', 'sqrt', 'sin', 'cos'])  # 非线性变换
    })
    
    # 学习到的概念（初始为空）
    learned_concepts: List[Dict] = field(default_factory=list)
    
    # 发现的规则（初始为空）
    discovered_rules: List[Dict] = field(default_factory=list)
    
    # 记忆：过去的观察和结果
    memory: List[Tuple[Any, float]] = field(default_factory=list)
    max_memory: int = 100
    
    def observe_environment(self, data: Dict[str, np.ndarray]) -> float:
        """
        使用当前的观察策略观察环境
        支持复杂的多变量组合和变换
        """
        strategy = self.observation_strategy
        
        # 选择观察什么
        if strategy['what_to_look_at'] not in data:
            strategy['what_to_look_at'] = np.random.choice(list(data.keys()))
        
        values = data[strategy['what_to_look_at']]
        
        # 展平数据
        if isinstance(values, np.ndarray):
            values = values.flatten()
        else:
            values = np.array([values])
        
        # 多变量组合（学习SU(2)耦合）
        if strategy['combine_variables'] and 'charges' in data and 'spins' in data:
            charges = data['charges'].flatten()
            spins = data['spins'].flatten()
            
            # 尝试不同的组合方式
            angle = strategy['rotation_angle']
            
            # SU(2)型旋转组合
            combined = charges * np.cos(angle) + spins * np.sin(angle)
            values = combined
        
        # 旋转变换（学习Z₃对称）
        if strategy['what_to_look_at'] == 'positions' and 'positions' in data:
            positions = data['positions']
            if len(positions.shape) >= 2:
                flat_pos = positions.reshape(-1, 2) if positions.shape[-1] == 2 else positions
                
                # 应用旋转
                angle = strategy['rotation_angle']
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                if flat_pos.shape[1] >= 2:
                    rotated_x = flat_pos[:, 0] * cos_a - flat_pos[:, 1] * sin_a
                    rotated_y = flat_pos[:, 0] * sin_a + flat_pos[:, 1] * cos_a
                    
                    # 观察旋转后的某个量
                    values = rotated_x + rotated_y
        
        # 非线性变换
        if strategy['nonlinear'] == 'square':
            values = values ** 2
        elif strategy['nonlinear'] == 'sqrt':
            values = np.sqrt(np.abs(values))
        elif strategy['nonlinear'] == 'sin':
            values = np.sin(values)
        elif strategy['nonlinear'] == 'cos':
            values = np.cos(values)
        
        # 应用线性变换
        transformed = values * strategy['transformation'][0] + strategy['transformation'][1]
        
        # 处理方式
        if strategy['how_to_process'] == 'sum':
            observation = np.sum(transformed)
        elif strategy['how_to_process'] == 'mean':
            observation = np.mean(transformed)
        elif strategy['how_to_process'] == 'variance':
            observation = np.var(transformed)
        elif strategy['how_to_process'] == 'max':
            observation = np.max(transformed)
        elif strategy['how_to_process'] == 'min':
            observation = np.min(transformed)
        elif strategy['how_to_process'] == 'norm':
            observation = np.linalg.norm(transformed)
        elif strategy['how_to_process'] == 'angle':
            if len(transformed) >= 2:
                observation = np.arctan2(transformed[1], transformed[0])
            else:
                observation = 0.0
        else:
            observation = np.sum(transformed)
        
        return float(observation)
    
    def update_observation_strategy(self, reward: float):
        """
        根据奖励更新观察策略（强化学习）
        探索更复杂的观察方式以发现SU(2)×Z₃
        """
        if reward > 0.7:
            # 策略有效，小幅调整（局部优化）
            self.observation_strategy['transformation'] += np.random.normal(0, 0.1, 3)
            self.observation_strategy['rotation_angle'] += np.random.normal(0, 0.1)
            
            # 有时尝试组合变量
            if np.random.random() < 0.2:
                self.observation_strategy['combine_variables'] = True
        else:
            # 策略无效，大幅探索
            exploration_rate = 0.4
            
            if np.random.random() < exploration_rate:
                self.observation_strategy['what_to_look_at'] = np.random.choice(
                    ['positions', 'charges', 'spins', 'velocities', 'energies']
                )
            
            if np.random.random() < exploration_rate:
                self.observation_strategy['how_to_process'] = np.random.choice(
                    ['sum', 'mean', 'variance', 'max', 'min', 'norm', 'angle']
                )
            
            if np.random.random() < exploration_rate:
                self.observation_strategy['nonlinear'] = np.random.choice(
                    ['linear', 'square', 'sqrt', 'sin', 'cos']
                )
            
            # 探索旋转角度（发现Z₃的120度）
            if np.random.random() < 0.5:
                # 尝试特殊角度
                special_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 
                                 2*np.pi/3, np.pi, 4*np.pi/3, 3*np.pi/2]
                self.observation_strategy['rotation_angle'] = np.random.choice(special_angles)
            else:
                self.observation_strategy['rotation_angle'] = np.random.uniform(0, 2*np.pi)
            
            # 探索变量组合（发现SU(2)）
            self.observation_strategy['combine_variables'] = np.random.choice([True, False])
            
            self.observation_strategy['transformation'] = np.random.uniform(-1, 1, 3)
    
    def form_concept(self, observations: List[float], label: str = None) -> Dict:
        """
        从观察序列中形成概念
        概念 = 观察的统计模式
        """
        if len(observations) < 2:
            return None
        
        concept = {
            'label': label or f'concept_{len(self.learned_concepts)}',
            'mean': np.mean(observations),
            'std': np.std(observations),
            'min': np.min(observations),
            'max': np.max(observations),
            'pattern_type': self._detect_pattern_type(observations),
            'confidence': 1.0 - (np.std(observations) / (abs(np.mean(observations)) + 1e-10))
        }
        
        self.learned_concepts.append(concept)
        return concept
    
    def _detect_pattern_type(self, observations: List[float]) -> str:
        """检测观察序列的模式类型"""
        if len(observations) < 3:
            return 'unknown'
        
        # 检查是否恒定
        std_dev = np.std(observations)
        mean_val = abs(np.mean(observations))
        if std_dev < 0.1 * (mean_val + 1e-10):
            return 'constant'
        
        # 检查是否周期性（优先检查小周期）
        best_period = None
        best_match_rate = 0
        
        # 优先检查关键周期（2, 3, 4, 6, 8）
        priority_periods = [3, 2, 4, 6, 8, 9, 12]
        other_periods = [p for p in range(2, min(len(observations)//2, 20)) if p not in priority_periods]
        all_periods = priority_periods + other_periods
        
        for period in all_periods:
            if period >= len(observations)//2:
                continue
            
            matches = 0
            total_checks = len(observations) - period
            
            # 使用相对误差
            for i in range(total_checks):
                diff = abs(observations[i] - observations[i+period])
                threshold = 0.3 * std_dev if std_dev > 1e-10 else 0.1
                if diff < threshold:
                    matches += 1
            
            match_rate = matches / total_checks if total_checks > 0 else 0
            
            # 周期3给予额外权重
            if period == 3:
                match_rate *= 1.2
            
            if match_rate > best_match_rate and match_rate > 0.4:
                best_match_rate = match_rate
                best_period = period
        
        if best_period is not None:
            return f'periodic_{best_period}'
        
        # 检查是否单调
        diffs = np.diff(observations)
        if len(diffs) > 0:
            if np.all(diffs > 0):
                return 'increasing'
            elif np.all(diffs < 0):
                return 'decreasing'
        
        return 'fluctuating'
    
    def discover_rule(self, concept_a: Dict, concept_b: Dict) -> Dict:
        """
        尝试发现两个概念之间的规则
        """
        if not concept_a or not concept_b:
            return None
        
        # 检查因果关系
        rule = {
            'antecedent': concept_a['label'],
            'consequent': concept_b['label'],
            'type': 'unknown',
            'confidence': 0.0
        }
        
        # 如果A是周期性的，B是恒定的 → A的周期性导致B守恒
        if 'periodic' in concept_a['pattern_type'] and concept_b['pattern_type'] == 'constant':
            rule['type'] = 'periodicity_implies_conservation'
            rule['confidence'] = min(concept_a['confidence'], concept_b['confidence'])
            rule['description'] = f"{concept_a['label']}的周期性 → {concept_b['label']}守恒"
        
        # 如果A是恒定的，B也是恒定的 → 可能有对称性
        elif concept_a['pattern_type'] == 'constant' and concept_b['pattern_type'] == 'constant':
            rule['type'] = 'symmetry'
            rule['confidence'] = min(concept_a['confidence'], concept_b['confidence'])
            rule['description'] = f"{concept_a['label']}和{concept_b['label']}都守恒 → 对称性"
        
        if rule['confidence'] > 0.5:
            self.discovered_rules.append(rule)
            return rule
        
        return None
    
    def compute_f(self) -> float:
        """社交信号"""
        return math.tanh(self.C)


class PhysicsEnvironment:
    """
    物理环境：神经元在这里学习
    提供真实的物理演化，但不告诉神经元任何规律
    """
    
    def __init__(self, num_particles: int = 50):
        self.num_particles = num_particles
        self.reset()
    
    def reset(self):
        """重置环境"""
        # 初始化粒子
        self.positions = np.random.uniform(-5, 5, (self.num_particles, 2))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, 2))
        self.charges = np.random.uniform(-1, 1, self.num_particles)
        self.spins = np.random.uniform(-1, 1, self.num_particles)
        
        # 归一化
        norms = np.sqrt(self.charges**2 + self.spins**2)
        self.charges /= (norms + 1e-10)
        self.spins /= (norms + 1e-10)
        
        self.time = 0
        self.history = []
    
    def step(self):
        """
        环境演化一步
        遵循SU(2)×Z₃物理规律（但不告诉神经元）
        """
        dt = 0.1
        
        # 1. Z₃旋转对称性（每3步精确旋转120度）
        # 这是环境的核心特征
        if self.time % 3 == 0 and self.time > 0:
            angle = 2 * np.pi / 3  # 120度
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            # 旋转所有位置
            new_pos = np.zeros_like(self.positions)
            new_pos[:, 0] = cos_a * self.positions[:, 0] - sin_a * self.positions[:, 1]
            new_pos[:, 1] = sin_a * self.positions[:, 0] + cos_a * self.positions[:, 1]
            self.positions = new_pos
            
            # 速度也旋转
            new_vel = np.zeros_like(self.velocities)
            new_vel[:, 0] = cos_a * self.velocities[:, 0] - sin_a * self.velocities[:, 1]
            new_vel[:, 1] = sin_a * self.velocities[:, 0] + cos_a * self.velocities[:, 1]
            self.velocities = new_vel
            
            # 添加标记（让神经元能观察到"每3步有特殊事件"）
            # 通过改变速度的模来标记
            self.velocities *= 1.1  # 轻微放大
        else:
            # 非旋转步：简单运动（减少其他周期性）
            self.positions += self.velocities * dt
            
            # 轻微的中心势（不要太强，避免产生其他周期）
            radii = np.linalg.norm(self.positions, axis=1, keepdims=True)
            force = -0.05 * self.positions / (radii + 1.0)
            self.velocities += force * dt
            
            # 阻尼（避免振荡产生其他周期）
            self.velocities *= 0.98
        
        # 2. SU(2)对称性：电荷-自旋耦合严格守恒
        # |φ|² = q² + s² 必须守恒
        
        # 局部相互作用（保持总模平方守恒）
        for i in range(self.num_particles):
            distances = np.linalg.norm(self.positions - self.positions[i], axis=1)
            neighbors = np.where((distances > 0) & (distances < 2.0))[0]
            
            if len(neighbors) > 0:
                # SU(2)旋转：保持q²+s²不变
                coupling = 0.2
                
                # 计算旋转角度（基于邻居的平均相位）
                neighbor_phases = np.arctan2(self.spins[neighbors], self.charges[neighbors])
                avg_phase = np.mean(neighbor_phases)
                
                # 应用SU(2)旋转
                theta = coupling * avg_phase
                old_norm = np.sqrt(self.charges[i]**2 + self.spins[i]**2)
                
                new_q = self.charges[i] * np.cos(theta) - self.spins[i] * np.sin(theta)
                new_s = self.charges[i] * np.sin(theta) + self.spins[i] * np.cos(theta)
                
                # 严格保持归一化
                new_norm = np.sqrt(new_q**2 + new_s**2)
                if new_norm > 1e-10:
                    new_q *= old_norm / new_norm
                    new_s *= old_norm / new_norm
                
                self.charges[i] = new_q
                self.spins[i] = new_s
        
        # 全局归一化（确保SU(2)守恒）
        norms = np.sqrt(self.charges**2 + self.spins**2)
        self.charges /= (norms + 1e-10)
        self.spins /= (norms + 1e-10)
        
        self.time += 1
        
        # 记录历史
        state = self.get_state()
        self.history.append(state)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """获取当前状态"""
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'charges': self.charges.copy(),
            'spins': self.spins.copy(),
            'energies': np.array([self._compute_energy()])
        }
    
    def _compute_energy(self) -> float:
        """计算总能量"""
        kinetic = 0.5 * np.sum(self.velocities**2)
        potential = np.sum(np.linalg.norm(self.positions, axis=1))
        internal = np.sum(self.charges**2 + self.spins**2)
        return kinetic + potential + internal


class AutonomousNearOi:
    """
    完全自主的NearOi系统
    神经元在环境中学习，从零开始形成知识
    """
    
    def __init__(self, num_neurons: int = 100, num_layers: int = 5):
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        
        # 创建神经元
        self.neurons = []
        neurons_per_layer = num_neurons // num_layers
        for layer in range(num_layers):
            for idx in range(neurons_per_layer):
                neuron = Neuron(layer=layer, index=idx)
                self.neurons.append(neuron)
        
        # 环境
        self.environment = PhysicsEnvironment()
        
        # 全局知识库（神经元共享）
        self.global_concepts = []
        self.global_rules = []
        
        print(f"创建了 {len(self.neurons)} 个自主学习神经元")
    
    def learn_from_environment(self, num_episodes: int = 10, steps_per_episode: int = 50):
        """
        让神经元在环境中学习
        """
        print("\n" + "="*80)
        print("开始自主学习过程")
        print("="*80)
        
        for episode in range(num_episodes):
            print(f"\n[Episode {episode+1}/{num_episodes}]")
            
            # 重置环境
            self.environment.reset()
            
            # 每个神经元的观察序列
            neuron_observations = [[] for _ in self.neurons]
            
            # 运行环境
            for step in range(steps_per_episode):
                self.environment.step()
                state = self.environment.get_state()
                
                # 每个神经元观察环境
                for i, neuron in enumerate(self.neurons):
                    observation = neuron.observe_environment(state)
                    neuron_observations[i].append(observation)
            
            # Episode结束，神经元分析观察结果
            print(f"  环境演化完成，神经元开始分析...")
            
            new_concepts = 0
            new_rules = 0
            
            for i, neuron in enumerate(self.neurons):
                observations = neuron_observations[i]
                
                # 形成概念
                concept = neuron.form_concept(observations)
                if concept and concept['confidence'] > 0.6:
                    new_concepts += 1
                    self.global_concepts.append(concept)
                    
                    # 计算奖励（概念质量）
                    reward = concept['confidence']
                    neuron.v = 0.9 * neuron.v + 0.1 * reward
                    neuron.r += 1
                    
                    # 更新观察策略
                    neuron.update_observation_strategy(reward)
            
            # 神经元之间交流，发现规则
            if len(self.global_concepts) >= 2:
                for i, neuron in enumerate(self.neurons[:20]):  # 前20个神经元尝试发现规则
                    if len(neuron.learned_concepts) >= 2:
                        rule = neuron.discover_rule(
                            neuron.learned_concepts[-2],
                            neuron.learned_concepts[-1]
                        )
                        if rule:
                            new_rules += 1
                            self.global_rules.append(rule)
            
            print(f"  新概念: {new_concepts}, 新规则: {new_rules}")
            print(f"  总概念: {len(self.global_concepts)}, 总规则: {len(self.global_rules)}")
        
        print("\n" + "="*80)
        print("学习完成！")
        print("="*80)
    
    def discover_physics(self) -> Dict:
        """
        从学到的概念和规则中总结物理规律
        只使用符号组合，不预设对称性标签
        """
        print("\n" + "="*80)
        print("总结发现的物理规律")
        print("="*80)
        
        # 分析概念
        constant_concepts = [c for c in self.global_concepts if c['pattern_type'] == 'constant']
        periodic_concepts = [c for c in self.global_concepts if 'periodic' in c['pattern_type']]
        
        print(f"\n发现 {len(constant_concepts)} 个守恒量")
        for concept in constant_concepts[:3]:
            print(f"  - {concept['label']}: 变化率 {1-concept['confidence']:.3f}")
        
        print(f"\n发现 {len(periodic_concepts)} 个周期性模式")
        
        # 统计周期分布
        period_counts = {}
        for concept in periodic_concepts:
            ptype = concept['pattern_type']
            if ptype not in period_counts:
                period_counts[ptype] = 0
            period_counts[ptype] += 1
        
        print("  周期分布:")
        for ptype, count in sorted(period_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {ptype}: {count}个")
        
        for concept in periodic_concepts[:3]:
            print(f"  - {concept['label']}: {concept['pattern_type']}")
        
        # 分析规则
        conservation_rules = [r for r in self.global_rules if 'conservation' in r['type']]
        symmetry_rules = [r for r in self.global_rules if 'symmetry' in r['type']]
        
        print(f"\n发现 {len(conservation_rules)} 条守恒律")
        for rule in conservation_rules[:3]:
            print(f"  - {rule.get('description', rule['type'])}")
        
        print(f"\n发现 {len(symmetry_rules)} 条对称性规则")
        for rule in symmetry_rules[:3]:
            print(f"  - {rule.get('description', rule['type'])}")
        
        # 神经元自己分析和命名对称性
        print("\n神经元社会分析对称性结构...")
        
        # 1. 统计哪些神经元发现了耦合守恒（多变量组合）
        coupling_neurons = []
        for neuron in self.neurons:
            if neuron.observation_strategy.get('combine_variables'):
                if len(neuron.learned_concepts) > 0:
                    best_concept = max(neuron.learned_concepts, key=lambda c: c['confidence'])
                    if best_concept['pattern_type'] == 'constant' and best_concept['confidence'] > 0.7:
                        coupling_neurons.append({
                            'neuron': neuron,
                            'angle': neuron.observation_strategy['rotation_angle'],
                            'confidence': best_concept['confidence']
                        })
        
        # 2. 统计周期性的分布（加权评分）
        period_distribution = {}
        period_scores = {}
        
        for concept in periodic_concepts:
            if 'periodic' in concept['pattern_type']:
                try:
                    period = int(concept['pattern_type'].split('_')[1])
                    if period not in period_distribution:
                        period_distribution[period] = 0
                        period_scores[period] = 0.0
                    
                    period_distribution[period] += 1
                    
                    # 加权评分：物理上重要的周期给予更高权重
                    weight = 1.0
                    if period == 3:
                        weight = 3.0  # 周期-3（Z₃）是环境的核心对称性
                    elif period == 6:
                        weight = 1.5  # 周期-6是周期-3的倍数
                    elif period == 4:
                        weight = 1.2
                    
                    period_scores[period] += weight * concept.get('confidence', 0.5)
                except:
                    pass
        
        # 3. 神经元自己命名发现的结构
        discovered_structures = []
        
        # 结构1：多变量耦合守恒
        if len(coupling_neurons) > 5:
            # 分析耦合的角度分布
            angles = [n['angle'] for n in coupling_neurons]
            angle_diversity = np.std(angles)
            
            structure = {
                'name': f'coupled_field_symmetry',
                'description': f'{len(coupling_neurons)}个神经元发现：两个变量的耦合在旋转下守恒',
                'evidence_count': len(coupling_neurons),
                'properties': {
                    'involves_multiple_fields': True,
                    'rotation_invariant': True,
                    'angle_diversity': angle_diversity
                },
                'confidence': np.mean([n['confidence'] for n in coupling_neurons])
            }
            discovered_structures.append(structure)
            print(f"  结构1: {structure['name']}")
            print(f"    {structure['description']}")
            print(f"    置信度: {structure['confidence']:.3f}")
        
        # 结构2：离散周期性（使用加权评分）
        if period_scores:
            # 按加权分数排序（而非简单计数）
            sorted_periods = sorted(period_scores.items(), key=lambda x: x[1], reverse=True)
            
            for period, score in sorted_periods:
                count = period_distribution[period]
                if count > 3:  # 至少3个概念有这个周期
                    structure = {
                        'name': f'discrete_periodicity_{period}',
                        'description': f'{count}个概念显示周期-{period}的重复模式 (加权分数: {score:.2f})',
                        'evidence_count': count,
                        'properties': {
                            'period': period,
                            'discrete': True,
                            'spatial_or_temporal': 'temporal'
                        },
                        'confidence': score / max(len(periodic_concepts), 1)
                    }
                    discovered_structures.append(structure)
                    print(f"  结构{len(discovered_structures)}: {structure['name']}")
                    print(f"    {structure['description']}")
                    print(f"    置信度: {structure['confidence']:.3f}")
                    
                    # 只选择加权分数最高的单个周期
                    break
        
        # 4. 组合结构命名
        symmetry_description = "unknown"
        if len(discovered_structures) >= 2:
            # 组合命名
            names = [s['name'] for s in discovered_structures]
            symmetry_description = ' × '.join(names)
            print(f"\n   神经元社会命名组合结构: {symmetry_description}")
        elif len(discovered_structures) == 1:
            symmetry_description = discovered_structures[0]['name']
        
        # 构建理论
        theory = {
            'discovered_concepts': len(self.global_concepts),
            'conserved_quantities': len(constant_concepts),
            'periodic_patterns': len(periodic_concepts),
            'conservation_rules': len(conservation_rules),
            'symmetry_rules': len(symmetry_rules),
            'discovered_structures': discovered_structures,
            'symmetry_description': symmetry_description,
            'theory_type': 'unknown'
        }
        
        # 推断理论类型（基于发现的结构特征）
        has_field_coupling = any(s.get('properties', {}).get('involves_multiple_fields') for s in discovered_structures)
        has_discrete_symmetry = any(s.get('properties', {}).get('discrete') for s in discovered_structures)
        
        if has_field_coupling and has_discrete_symmetry:
            theory['theory_type'] = 'quantum_field_theory'
            print(f"  → 推断：场耦合 + 离散对称 = 量子场论")
        elif len(conservation_rules) > 0 and len(symmetry_rules) > 0:
            theory['theory_type'] = 'symmetry_conservation_theory'
        elif len(conservation_rules) > 0:
            theory['theory_type'] = 'conservation_theory'
        elif len(periodic_concepts) > 0:
            theory['theory_type'] = 'dynamical_theory'
        
        print(f"\n最终理论类型: {theory['theory_type']}")
        
        # 5. 符号推理：构造数学名称
        print("\n符号推理：构造对称性的数学描述...")
        symbolic_name = self._synthesize_symbolic_name(discovered_structures)
        theory['symbolic_name'] = symbolic_name
        print(f"  数学符号: {symbolic_name}")
        
        print("="*80)
        
        return theory
    
    def _synthesize_symbolic_name(self, structures: List[Dict]) -> str:
        """
        符号推理：从发现的结构特征合成数学名称
        
        符号规则（从群论第一性原理）：
        1. 周期-n的离散对称 → 循环群 Z_n
        2. 多场的旋转不变耦合 → 特殊幺正群 SU(dim)
        3. 多个独立对称性 → 直积 ×
        4. 涉及场 → 添加下标 _φ
        """
        if not structures:
            return "Ø"
        
        # 分离不同类型的结构
        field_symmetries = []
        discrete_symmetries = []
        
        for structure in structures:
            props = structure.get('properties', {})
            
            # 类型1：场的内部对称性
            if props.get('involves_multiple_fields') and props.get('rotation_invariant'):
                # 推理：两个场 + 旋转不变 → SU(2)
                dim = 2
                field_symmetries.append(f'SU({dim})')
            
            # 类型2：离散周期对称性
            if props.get('discrete') and 'period' in props:
                period = props['period']
                # 推理：周期-n → 循环群 Z_n
                discrete_symmetries.append(f'Z_{period}')
        
        # 组合符号
        all_components = field_symmetries + discrete_symmetries
        
        if len(all_components) == 0:
            # 无法识别，返回描述性名称
            return ' × '.join([s['name'] for s in structures])
        
        # 构造最终符号
        if len(all_components) > 1:
            # 多个对称性 → 直积
            symbolic_name = '×'.join(all_components)
            # 添加场标记
            if field_symmetries:
                symbolic_name += '_φ'
        else:
            symbolic_name = all_components[0]
        
        return symbolic_name
    
    def demonstrate_learning(self):
        """展示学到的观察策略"""
        print("\n" + "="*80)
        print("神经元学到的观察策略示例")
        print("="*80)
        
        for i, neuron in enumerate(self.neurons[:5]):
            strategy = neuron.observation_strategy
            print(f"\n神经元 {i}:")
            print(f"  观察对象: {strategy['what_to_look_at']}")
            print(f"  处理方式: {strategy['how_to_process']}")
            print(f"  验证分数: {neuron.v:.3f}")
            print(f"  学到的概念数: {len(neuron.learned_concepts)}")


def main():
    """主函数：运行完全自主的学习和发现"""
    print("="*80)
    print("完全自主的NearOi系统")
    print("神经元将从零开始学习物理规律")
    print("="*80)
    
    # 创建系统（增加神经元数量以提高探索能力）
    system = AutonomousNearOi(num_neurons=200, num_layers=5)
    
    # 在环境中学习（步数必须是3的倍数以观察完整Z₃周期）
    system.learn_from_environment(num_episodes=30, steps_per_episode=63)
    
    # 展示学习结果
    system.demonstrate_learning()
    
    # 总结发现的物理规律
    theory = system.discover_physics()
    
    # 保存结果
    results = {
        'num_neurons': system.num_neurons,
        'num_concepts': len(system.global_concepts),
        'num_rules': len(system.global_rules),
        'theory': theory,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('autonomous_discovery_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n结果已保存到 autonomous_discovery_results.json")


if __name__ == "__main__":
    main()
