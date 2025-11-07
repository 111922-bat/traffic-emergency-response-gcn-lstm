# GCN图卷积网络模块

基于最新研究结果实现的完整GCN图卷积网络模块，支持多种图卷积操作、动态邻接矩阵学习、多头注意力机制等功能。

## 功能特性

### 🏗️ 核心架构
- **多种图卷积操作**: ChebNet、GraphSAGE、GAT、基础GCN
- **动态邻接矩阵学习**: 支持端到端学习时变图结构
- **多头注意力机制**: 空间注意力和时间注意力
- **图结构数据预处理**: 完整的数据处理流水线

### 🔧 技术实现
- **PyTorch实现**: 基于PyTorch的完整实现
- **模块化设计**: 高度模块化，易于扩展
- **训练和推理**: 完整的训练和推理流程
- **性能评估**: 多种评估指标和可视化

### 📊 支持的功能
- 静态和动态图构建
- 多种归一化方法
- 滑动窗口数据生成
- 早停和学习率调度
- 模型保存和加载
- 结果可视化

## 文件结构

```
code/models/
├── gcn_network.py      # 主要实现文件
├── test_gcn.py         # 测试代码
├── gcn_examples.py     # 使用示例
└── README.md          # 本文档
```

## 快速开始

### 1. 基础使用

```python
from gcn_network import (
    GraphDataProcessor, GCNNetwork, GCNTrainer, 
    GCNEvaluator, create_sample_data
)

# 创建示例数据
data, coordinates = create_sample_data(
    n_timesteps=500, n_nodes=30, n_features=1
)

# 数据预处理
processor = GraphDataProcessor(normalization='zscore')
graph_data = processor.prepare_graph_data(
    data, coordinates, window_size=12, prediction_steps=3
)

# 创建模型
model = GCNNetwork(
    n_nodes=graph_data['n_nodes'],
    n_features=graph_data['n_features'],
    n_hidden=64,
    n_layers=3,
    conv_type='cheb',  # 可选: 'gcn', 'cheb', 'sage', 'gat'
    use_attention=True,
    use_dynamic_adj=True
)

# 训练模型
trainer = GCNTrainer(model, learning_rate=0.001)
training_history = trainer.train(train_loader, val_loader, graph_data_tensor)

# 评估模型
evaluator = GCNEvaluator(model)
results = evaluator.evaluate(test_loader, graph_data_tensor)
```

### 2. 交通流预测示例

```python
# 模拟城市路网
coordinates = np.random.uniform(0, 1000, (50, 2))  # 50个节点
adj_matrix = create_road_network(coordinates)      # 路网邻接矩阵

# 生成交通数据（速度、流量、占用率）
traffic_data = generate_traffic_data(
    n_timesteps=168*7, n_nodes=50, adj_matrix=adj_matrix
)

# 使用GraphSAGE进行预测
model = GCNNetwork(
    n_nodes=50,
    n_features=3,  # 速度、流量、占用率
    conv_type='sage',
    use_attention=True,
    use_dynamic_adj=True
)
```

## API文档

### GraphDataProcessor

图数据预处理器，负责数据预处理和图构建。

```python
processor = GraphDataProcessor(
    normalization='zscore',  # 归一化方法
    adj_threshold=0.1,       # 邻接矩阵阈值
    sigma2=1.0,             # 高斯核参数
    epsilon=0.1             # 距离阈值参数
)
```

主要方法：
- `build_adjacency_matrix()`: 构建邻接矩阵
- `normalize_data()`: 数据归一化
- `create_sliding_window()`: 创建滑动窗口
- `prepare_graph_data()`: 准备完整的图数据

### GCNNetwork

完整的GCN网络实现。

```python
model = GCNNetwork(
    n_nodes=50,                    # 节点数
    n_features=3,                  # 输入特征维度
    n_hidden=64,                   # 隐藏层维度
    n_layers=3,                    # GCN层数
    conv_type='cheb',              # 卷积类型
    prediction_steps=3,            # 预测步数
    use_attention=True,            # 使用注意力机制
    use_dynamic_adj=True,          # 使用动态邻接矩阵
    dropout=0.1                    # Dropout概率
)
```

支持的卷积类型：
- `'gcn'`: 基础图卷积网络
- `'cheb'`: 切比雪夫图卷积网络
- `'sage'`: GraphSAGE
- `'gat'`: 图注意力网络

### GCNTrainer

模型训练器。

```python
trainer = GCNTrainer(
    model=model,
    learning_rate=0.001,
    weight_decay=1e-5
)

# 训练模型
training_history = trainer.train(
    train_loader, val_loader, graph_data_tensor,
    epochs=100, patience=20
)
```

### GCNEvaluator

模型评估器。

```python
evaluator = GCNEvaluator(model)

# 评估模型
results = evaluator.evaluate(test_loader, graph_data_tensor)

# 可视化结果
evaluator.plot_results(results, save_path='results.png')
```

## 使用示例

### 运行测试

```bash
cd code/models
python test_gcn.py
```

### 运行示例

```bash
cd code/models
python gcn_examples.py
```

这将运行以下示例：
1. 基础使用示例
2. 交通流预测示例
3. 模型配置对比示例
4. 模型保存和加载示例

## 数据格式

### 输入数据格式

- **时序数据**: `[n_timesteps, n_nodes, n_features]`
- **节点坐标**: `[n_nodes, 2]`
- **邻接矩阵**: `[n_nodes, n_nodes]`

### 滑动窗口格式

- **输入序列**: `[n_samples, window_size, n_nodes, n_features]`
- **目标序列**: `[n_samples, prediction_steps, n_nodes, n_features]`

## 性能指标

支持的评估指标：
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数

## 配置选项

### 模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `n_nodes` | 节点数 | 必需 |
| `n_features` | 输入特征维度 | 必需 |
| `n_hidden` | 隐藏层维度 | 64 |
| `n_layers` | GCN层数 | 3 |
| `conv_type` | 卷积类型 | 'cheb' |
| `prediction_steps` | 预测步数 | 1 |
| `use_attention` | 使用注意力 | True |
| `use_dynamic_adj` | 动态邻接矩阵 | True |
| `dropout` | Dropout概率 | 0.1 |

### 训练配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `learning_rate` | 学习率 | 0.001 |
| `weight_decay` | 权重衰减 | 1e-5 |
| `epochs` | 训练轮数 | 100 |
| `patience` | 早停耐心值 | 20 |
| `batch_size` | 批大小 | 32 |

## 高级功能

### 动态邻接矩阵学习

模型可以学习时变的图结构：

```python
model = GCNNetwork(
    use_dynamic_adj=True,  # 启用动态邻接矩阵学习
    # ... 其他参数
)
```

### 多头注意力机制

支持空间和时间注意力：

```python
model = GCNNetwork(
    use_attention=True,    # 启用注意力机制
    # ... 其他参数
)
```

### 自定义图卷积层

可以自定义图卷积层：

```python
from gcn_network import ChebConv, GraphSAGEConv, GATConv

# 切比雪夫卷积
cheb_conv = ChebConv(in_features=64, out_features=128, k=3)

# GraphSAGE卷积
sage_conv = GraphSAGEConv(in_features=64, out_features=128, aggregator='mean')

# 图注意力卷积
gat_conv = GATConv(in_features=64, out_features=128, n_heads=8)
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减小批大小 `batch_size`
   - 减小隐藏层维度 `n_hidden`
   - 减小图规模 `n_nodes`

2. **训练不收敛**
   - 调整学习率 `learning_rate`
   - 检查数据归一化
   - 增加正则化 `dropout`

3. **梯度爆炸**
   - 使用梯度裁剪
   - 减小学习率
   - 增加 `weight_decay`

### 调试技巧

1. **检查数据形状**
   ```python
   print(f"输入数据形状: {X.shape}")
   print(f"邻接矩阵形状: {adj_matrix.shape}")
   ```

2. **监控训练过程**
   ```python
   # 在训练循环中添加
   if epoch % 10 == 0:
       print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
   ```

3. **可视化结果**
   ```python
   evaluator.plot_results(results)
   ```

## 扩展开发

### 添加新的图卷积层

```python
class MyCustomConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 自定义实现
    
    def forward(self, x, adj_matrix):
        # 前向传播逻辑
        return out
```

### 添加新的注意力机制

```python
class MyCustomAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 自定义实现
    
    def forward(self, query, key, value):
        # 注意力计算逻辑
        return output, attention_weights
```

## 性能优化

### 训练优化
- 使用更大的批大小
- 启用混合精度训练
- 使用分布式训练
- 启用梯度累积

### 推理优化
- 模型量化
- 模型剪枝
- ONNX导出
- TensorRT加速

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 更新日志

### v1.0.0 (2025-11-05)
- 初始版本发布
- 支持多种图卷积操作
- 实现动态邻接矩阵学习
- 添加多头注意力机制
- 完整的训练和评估流程

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本实现基于最新的图神经网络研究结果，适用于交通流预测、社交网络分析、推荐系统等多种图数据应用场景。