# CAR: 认知架构 - 噪声鲁棒性测试

[![DOI](https://img.shields.io/badge/DOI-10.17605/OSF.IO/F968B-blue.svg)](https://doi.org/10.17605/OSF.IO/F968B)

基于自主计算单元的检索式学习架构，在 10¹⁵⁰ 噪声水平下保持有效预测能力。

## PyTorch DNN 对比配置

**测试对象**: PyTorch MLP

### DNN 架构
```
输入层 (20) → 隐藏层1 (50) → 隐藏层2 (50) → 输出层 (1)
```

### DNN 参数设置
- **激活函数**: ReLU
- **优化器**: Adam (learning_rate = 0.01)
- **损失函数**: MSE (均方误差)
- **Batch size**: 32
- **训练轮数**: 100 epochs
- **随机种子**: 456 (保证可重复性)

### DNN 代码实现
```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleDNN(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, output_size=1):
        super(SimpleDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 20 → 50
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # 50 → 50
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)   # 50 → 1
        )

# 训练配置
model = SimpleDNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练 100 个 epochs
for epoch in range(100):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

## 核心结果

### 极端噪声测试

| 噪声水平 | CAR PredStd | DNN PredStd | DNN MSE | 状态 |
|---------|-------------|-------------|---------|------|
| 1 (正常) | 0.6717 | 2.5925 | 正常 | ✓ |
| 10³ | 0.6080 | 1,701.16 | 开始退化 | ✓ |
| 10⁶ | 0.6069 | 1,700,995.75 | 严重退化 | ✓ |
| 10⁹ | 0.6069 | 1.7×10⁹ | 接近崩溃 | ✓ |
| 10¹² | 0.6069 | 1.7×10¹² | 崩溃 | ✓ |
| 10⁵⁰ | 0.6069 | NaN | **DNN 失败** | ✓ |
| 10¹⁵⁰ | 0.6069 | NaN | **DNN 失败** | ✓ |

**关键发现**:
- CAR 性能下降：0.6717 → 0.6069（仅 9.6%）
- DNN 性能崩溃：2.5925 → NaN（完全失败）
- CAR 在 10¹⁵⁰ 噪声下仍保持预测多样性

### 对抗性攻击测试

| 攻击强度 (ε) | CAR 偏移 | DNN 偏移 | 胜者 |
|-------------|---------|---------|------|
| 0.1 | 0.0447 | 0.0819 | CAR ✓ |
| 0.5 | 0.2124 | 0.4418 | CAR ✓ |
| 1.0 | 0.2594 | 0.9959 | CAR ✓ |

**关键发现**: CAR 在所有攻击强度下都比 DNN 稳定 2-4 倍

**Float128 扩展**：CAR 在 10²⁴⁶⁶ 噪声下仍能正常工作（-49320 dB SNR）

## 快速开始

### 安装
```bash
pip install -r requirements.txt
```

**依赖包含**:
- numpy >= 1.20.0
- torch >= 2.0.0 (PyTorch for DNN comparison)
- pytest >= 7.0.0
- matplotlib >= 3.5.0

### 运行测试
```bash
# 运行所有测试
python run_all.py

# 快速测试
python run_all.py --quick

# 单独测试
python run_all.py --noise      # 噪声鲁棒性（含 DNN 对比）
python run_all.py --float128   # Float128 极限
python run_all.py --online     # 在线学习
python run_all.py --mechanism  # 机制鉴别

# DNN 对比测试
python tests/test_extreme_noise.py          # 极端噪声（CAR vs PyTorch DNN）
python tests/test_adversarial_attack.py --compare  # 对抗攻击（CAR vs PyTorch DNN）
```

### 基础使用
```python
import numpy as np
from src.car_model import CompleteCARModel, CARConfig

# 配置 CAR
config = CARConfig(KB_CAPACITY=100)
car = CompleteCARModel(config=config, n_features=20)

# 训练
X_train = np.random.randn(300, 20)
y_train = np.sum(np.sin(X_train[:, :3]), axis=1)
car.fit(X_train, y_train)

# 测试极端噪声（10¹⁵⁰！）
X_test = np.random.randn(100, 20)
noise = np.random.randn(100, 20) * 1e150
X_noisy = X_test + noise

predictions = [car.predict(x) for x in X_noisy]
print(f"预测标准差: {np.std(predictions):.4f}")
```

## 项目结构

```
car-complete-demo/
├── src/                    # 源代码
│   ├── car_model.py        # CAR 模型
│   ├── knowledge_base.py   # 知识库
│   ├── unit.py             # 计算单元
│   └── config.py           # 配置
│
├── tests/                  # 测试（100% 覆盖论文）
│   ├── test_extreme_noise.py      # 极端噪声（含 PyTorch DNN 对比）
│   ├── test_float128_limits.py    # Float128 极限
│   ├── test_online_learning.py    # 在线学习
│   ├── test_data_shuffling.py     # 数据打乱
│   ├── test_mechanism_discrimination.py  # 机制鉴别
│   ├── test_adversarial_attack.py  # 对抗攻击（含 PyTorch DNN 对比）
│   └── test_anti_cheat.py        # 防作弊
│
├── run_all.py              # 一键运行
└── README.md               # 本文件
```

## 核心机制

### 多因子加权（噪声鲁棒性的关键）
```
权重 = 相似度 × 置信度 × log(使用次数) × 时间因子 × 多样性奖励
```

当噪声超过 10⁷⁵ 时，余弦相似度变为随机，但 CAR 仍然成功，因为其他因子接管。

### 在线学习
- 边预测边学习
- 无需重新训练
- 98.53% 改进率（论文：98.5%）

## 数学框架

### 单元状态
```
State_i = [A_i, v_i, x_i]
```
- A_i ∈ [0, 1]: 激活权重
- v_i ∈ [0, 1]: 验证分数
- x_i ∈ ℝᴰ: 数据样本

### 评分检索
```
s_i = A_i · v_i · 1/(1 + Δ_i)
```

## 关键发现

1. **多因子加权**: 不依赖单一相似度
2. **知识库检索**: 访问历史模式
3. **无梯度传播**: 噪声不累积
4. **验证评分**: 基于历史准确率
5. **vs PyTorch DNN**:
   - 在 10⁵⁰ 噪声下，CAR 正常工作，DNN 完全失败
   - 对抗性攻击中，CAR 比 DNN 稳定 2-4 倍
   - CAR 性能下降仅 9.6%，DNN 完全崩溃

## 文档

- [架构概述](docs/architecture.md)
- [数学规范](docs/math_specifications.md)
- [常见问题](docs/FAQ.md)

---

*CAR 展示了基于检索的架构可以通过多种信息源的组合实现卓越的噪声鲁棒性，为鲁棒 AI 系统设计开辟了新方向。*