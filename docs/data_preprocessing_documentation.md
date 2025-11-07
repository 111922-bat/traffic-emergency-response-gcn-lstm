# 数据预处理文档

## 1. 数据预处理概述

本文档详细记录了交通流量预测模型中使用的数据预处理流程，旨在确保数据处理的一致性、透明度和可复现性。

### 1.1 预处理目标

- **数据清洗**：去除异常值和噪声
- **特征标准化**：确保不同特征具有相同的尺度
- **时序特征构建**：创建用于时序预测的输入序列
- **图结构构建**：构建交通网络的空间拓扑结构
- **数据集分割**：将数据划分为训练集、验证集和测试集

## 2. 数据集来源

### 2.1 数据集信息

| 数据集名称 | 数据类型 | 时间跨度 | 采样频率 | 节点数 | 边数 |
|------------|----------|----------|----------|--------|------|
| METR-LA | 交通流量 | 4个月 | 5分钟 | 207 | ~1,400 |
| PEMS-BAY | 交通流量 | 6个月 | 5分钟 | 325 | ~3,800 |

### 2.2 数据文件格式

- **交通流量数据**：CSV格式，包含时间戳和每个传感器的流量数据
- **元数据文件**：CSV格式，包含传感器的地理坐标和其他属性
- **邻接矩阵**：基于传感器间距离或道路网络构建

## 3. 详细预处理步骤

### 3.1 数据加载

```python
# 加载原始交通流量数据
def load_traffic_data(data_dir, dataset_name):
    """
    加载交通流量数据
    
    参数:
        data_dir: 数据目录路径
        dataset_name: 数据集名称 ("METR-LA" 或 "PEMS-BAY")
    
    返回:
        加载的交通流量数据
    """
    # 根据数据集名称确定文件路径
    if dataset_name == "METR-LA":
        traffic_file = os.path.join(data_dir, "METR-LA.csv")
    elif dataset_name == "PEMS-BAY":
        traffic_file = os.path.join(data_dir, "PEMS-BAY.csv")
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 加载CSV文件
    traffic_data = pd.read_csv(traffic_file)
    return traffic_data
```

### 3.2 数据清洗

#### 3.2.1 缺失值处理

- **方法**：前向填充和后向填充结合
- **参数**：最大连续缺失值限制为3个时间步
- **实现**：

```python
# 处理缺失值
def handle_missing_values(data, max_consecutive_missing=3):
    """
    处理数据中的缺失值
    
    参数:
        data: 原始数据
        max_consecutive_missing: 最大连续缺失值限制
    
    返回:
        清洗后的数据
    """
    # 前向填充
    data_filled = data.fillna(method='ffill')
    # 后向填充处理前向填充无法解决的开头缺失值
    data_filled = data_filled.fillna(method='bfill')
    return data_filled
```

#### 3.2.2 异常值检测与处理

- **方法**：使用3σ法则检测异常值
- **参数**：标准差倍数 = 3.0
- **实现**：

```python
# 异常值检测与处理
def handle_outliers(data, std_threshold=3.0):
    """
    检测并处理异常值
    
    参数:
        data: 数据
        std_threshold: 标准差阈值，默认3个标准差
    
    返回:
        处理后的数据
    """
    # 计算每个节点的均值和标准差
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    
    # 检测异常值 (超出均值±3σ)
    lower_bound = means - std_threshold * stds
    upper_bound = means + std_threshold * stds
    
    # 将异常值限制在上下界内
    data_clipped = data.clip(lower=lower_bound, upper=upper_bound, axis=1)
    return data_clipped
```

### 3.3 特征标准化

- **方法**：Z-score标准化
- **参数**：使用训练集的均值和标准差
- **实现**：

```python
# 数据标准化
def standardize_data(train_data, val_data=None, test_data=None):
    """
    对数据进行Z-score标准化
    
    参数:
        train_data: 训练数据
        val_data: 验证数据 (可选)
        test_data: 测试数据 (可选)
    
    返回:
        标准化后的数据和标准化参数
    """
    # 使用训练集计算均值和标准差
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    
    # 避免除零错误
    std = std.replace(0, 1e-10)
    
    # 标准化数据
    train_normalized = (train_data - mean) / std
    
    results = {
        'train': train_normalized,
        'mean': mean,
        'std': std
    }
    
    # 标准化验证集和测试集（使用训练集的参数）
    if val_data is not None:
        results['val'] = (val_data - mean) / std
    
    if test_data is not None:
        results['test'] = (test_data - mean) / std
    
    return results
```

### 3.4 时序特征构建

- **方法**：滑动窗口构建输入输出序列
- **参数**：
  - `sequence_length`: 输入序列长度（默认为12，对应1小时数据）
  - `prediction_steps`: 预测步数（默认为3，对应15分钟）
- **实现**：

```python
# 构建时序特征
def build_sequences(data, sequence_length, prediction_steps):
    """
    构建用于训练和评估的输入输出序列
    
    参数:
        data: 标准化后的数据
        sequence_length: 输入序列长度
        prediction_steps: 预测步数
    
    返回:
        输入序列X和目标序列y
    """
    X, y = [], []
    
    # 遍历数据构建序列
    for i in range(len(data) - sequence_length - prediction_steps + 1):
        # 输入序列：过去的sequence_length个时间步
        X.append(data.iloc[i:i+sequence_length].values)
        # 目标序列：未来的prediction_steps个时间步
        y.append(data.iloc[i+sequence_length:i+sequence_length+prediction_steps].values)
    
    return np.array(X), np.array(y)
```

### 3.5 图结构构建

#### 3.5.1 基于距离的邻接矩阵

- **方法**：基于传感器之间的欧几里得距离构建邻接矩阵
- **参数**：
  - `distance_threshold`: 距离阈值（千米）
  - `normalization`: 归一化方法
- **实现**：

```python
# 构建基于距离的邻接矩阵
def build_distance_adj_matrix(coordinates, distance_threshold=1.0, normalization='minmax'):
    """
    基于传感器坐标构建邻接矩阵
    
    参数:
        coordinates: 传感器坐标数组 [nodes, 2]
        distance_threshold: 距离阈值（千米）
        normalization: 归一化方法 ('minmax' 或 'zscore')
    
    返回:
        邻接矩阵
    """
    num_nodes = len(coordinates)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    # 计算每对节点之间的距离
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # 计算欧几里得距离
                distance = np.sqrt(
                    (coordinates[i, 0] - coordinates[j, 0])**2 +
                    (coordinates[i, 1] - coordinates[j, 1])**2
                )
                
                # 转换为千米（假设坐标单位为度）
                distance_km = distance * 111.0  # 近似值，实际取决于纬度
                
                # 如果距离小于阈值，则认为有连接
                if distance_km <= distance_threshold:
                    adj_matrix[i, j] = 1.0 / (1.0 + distance_km)  # 距离衰减权重
    
    # 归一化
    if normalization == 'minmax' and adj_matrix.max() > 0:
        adj_matrix = adj_matrix / adj_matrix.max()
    elif normalization == 'zscore' and adj_matrix.std() > 0:
        adj_matrix = (adj_matrix - adj_matrix.mean()) / adj_matrix.std()
    
    return adj_matrix
```

#### 3.5.2 动态图构建

- **方法**：基于交通流量相关性动态调整邻接矩阵
- **参数**：
  - `correlation_threshold`: 相关性阈值
  - `update_interval`: 更新间隔（时间步）
- **实现**：

```python
# 构建动态邻接矩阵
def build_dynamic_adj_matrix(traffic_data, base_adj_matrix, correlation_threshold=0.5, window_size=24):
    """
    基于交通流量相关性构建动态邻接矩阵
    
    参数:
        traffic_data: 交通流量数据
        base_adj_matrix: 基础邻接矩阵
        correlation_threshold: 相关性阈值
        window_size: 计算相关性的窗口大小
    
    返回:
        动态邻接矩阵函数
    """
    def get_dynamic_adj(time_step):
        # 如果时间步太小，返回基础邻接矩阵
        if time_step < window_size:
            return base_adj_matrix
        
        # 计算最近窗口内的相关性
        recent_data = traffic_data.iloc[time_step-window_size:time_step].T
        correlation_matrix = np.corrcoef(recent_data)
        
        # 只保留显著相关的连接
        correlation_mask = np.abs(correlation_matrix) >= correlation_threshold
        
        # 结合基础邻接矩阵和相关性矩阵
        dynamic_adj = base_adj_matrix.copy()
        dynamic_adj[correlation_mask] = dynamic_adj[correlation_mask] * correlation_matrix[correlation_mask]
        
        # 归一化
        if dynamic_adj.max() > 0:
            dynamic_adj = dynamic_adj / dynamic_adj.max()
        
        return dynamic_adj
    
    return get_dynamic_adj
```

### 3.6 数据集分割

- **方法**：时间顺序分割
- **参数**：
  - `train_ratio`: 训练集比例（默认为0.7）
  - `val_ratio`: 验证集比例（默认为0.1）
  - `test_ratio`: 测试集比例（默认为0.2）
  - `random_seed`: 随机种子（确保可复现性，默认为42）
- **实现**：

```python
# 数据集分割
def split_dataset(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42):
    """
    将数据按时间顺序分割为训练集、验证集和测试集
    
    参数:
        data: 完整数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    
    返回:
        训练集、验证集和测试集
    """
    # 验证比例和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例和必须为1"
    
    # 设置随机种子确保可复现性
    np.random.seed(random_seed)
    
    # 计算分割点
    n_samples = len(data)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # 按时间顺序分割
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    test_data = data.iloc[train_size+val_size:]
    
    return train_data, val_data, test_data
```

## 4. 数据版本控制

### 4.1 数据版本管理方法

- **版本命名格式**：`v{major}.{minor}.{patch}`
- **版本号规则**：
  - `major`: 数据集结构或来源发生重大变化
  - `minor`: 预处理流程发生变化
  - `patch`: 预处理参数发生微调

### 4.2 版本信息记录

每个数据版本应记录以下信息：

```json
{
  "version": "v1.0.0",
  "created_at": "2023-10-20T14:30:00Z",
  "description": "初始数据版本",
  "datasets": ["METR-LA", "PEMS-BAY"],
  "preprocessing_steps": [
    {
      "step": "missing_value_handling",
      "params": {"max_consecutive_missing": 3}
    },
    {
      "step": "outlier_detection",
      "params": {"std_threshold": 3.0}
    },
    {
      "step": "normalization",
      "params": {"method": "zscore"}
    },
    {
      "step": "sequence_building",
      "params": {"sequence_length": 12, "prediction_steps": 3}
    }
  ],
  "split_ratio": {"train": 0.7, "val": 0.1, "test": 0.2},
  "random_seed": 42,
  "statistics": {
    "train_samples": 10080,
    "val_samples": 1440,
    "test_samples": 2880,
    "mean": 50.5,
    "std": 25.3
  }
}
```

## 5. 数据加载器配置

### 5.1 数据加载器参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `data_dir` | str | "../data/real-data" | 数据目录路径 |
| `dataset_name` | str | "METR-LA" | 数据集名称 |
| `sequence_length` | int | 12 | 输入序列长度 |
| `prediction_steps` | int | 3 | 预测步数 |
| `batch_size` | int | 32 | 批次大小 |
| `train_ratio` | float | 0.7 | 训练集比例 |
| `val_ratio` | float | 0.1 | 验证集比例 |
| `test_ratio` | float | 0.2 | 测试集比例 |
| `random_seed` | int | 42 | 随机种子 |
| `shuffle` | bool | True | 是否打乱训练数据 |

### 5.2 数据加载器使用示例

```python
# 初始化数据加载器
data_loader = RealTrafficDataLoader(
    data_dir="../data/real-data",
    dataset_name="METR-LA",
    sequence_length=12,
    prediction_steps=3
)

# 获取数据加载器
train_loader, val_loader, test_loader = data_loader.get_data_loaders(
    batch_size=32,
    shuffle=True,
    random_seed=42
)

# 获取邻接矩阵
adj_matrix = data_loader.build_adj_matrix()
```

## 6. 预处理流程执行

### 6.1 完整预处理流程

1. 加载原始交通流量数据
2. 加载元数据（传感器坐标）
3. 处理缺失值
4. 检测并处理异常值
5. 按时间顺序分割数据集（训练集/验证集/测试集）
6. 使用训练集参数对所有数据集进行标准化
7. 构建输入输出序列
8. 构建图结构（邻接矩阵）
9. 创建PyTorch数据加载器
10. 保存处理后的数据和预处理参数

### 6.2 预处理结果验证

预处理完成后，应验证以下内容：

- 无缺失值
- 无异常值
- 训练集、验证集和测试集比例正确
- 序列长度和预测步数正确
- 邻接矩阵维度正确
- 数据统计量合理

## 7. 常见问题与解决方案

### 7.1 数据质量问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 长时间连续缺失值 | 传感器故障 | 考虑移除该传感器或使用更复杂的插补方法 |
| 异常高流量值 | 传感器错误或特殊事件 | 使用3σ法则检测并限制异常值 |
| 数据分布不均衡 | 交通模式差异 | 考虑使用对数变换或分位数标准化 |

### 7.2 性能优化

- 对于大规模数据集，可以使用增量式加载
- 预处理结果可以缓存以避免重复计算
- 考虑使用并行计算加速数据处理

## 8. 扩展与维护

### 8.1 添加新数据集

添加新数据集时，需要：
1. 确保数据格式兼容
2. 更新数据加载函数
3. 根据新数据特点调整预处理参数
4. 记录新数据集的版本信息

### 8.2 预处理流程更新

当预处理流程发生变化时：
1. 更新相应的预处理函数
2. 增加版本号
3. 重新处理数据并保存新版本
4. 更新文档

## 9. 预处理脚本使用指南

### 9.1 运行预处理

```bash
# 运行预处理脚本
python code/data_integration/preprocess.py \
    --data_dir=data/real-data \
    --dataset_name=METR-LA \
    --output_dir=data/processed \
    --sequence_length=12 \
    --prediction_steps=3 \
    --train_ratio=0.7 \
    --random_seed=42
```

### 9.2 预处理参数说明

- `--data_dir`: 原始数据目录
- `--dataset_name`: 数据集名称（METR-LA 或 PEMS-BAY）
- `--output_dir`: 处理后数据保存目录
- `--sequence_length`: 输入序列长度
- `--prediction_steps`: 预测步数
- `--train_ratio`: 训练集比例
- `--val_ratio`: 验证集比例
- `--test_ratio`: 测试集比例
- `--random_seed`: 随机种子
- `--std_threshold`: 异常值检测的标准差阈值
- `--distance_threshold`: 邻接矩阵构建的距离阈值