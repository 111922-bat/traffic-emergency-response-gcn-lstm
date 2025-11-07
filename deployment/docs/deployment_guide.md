# 模型压缩和部署优化指南

## 概述

本指南提供了完整的模型压缩和部署优化解决方案，包括模型剪枝、知识蒸馏、量化优化、缓存策略、内存管理和部署架构优化。

## 目录结构

```
code/deployment/
├── compression/              # 模型压缩模块
│   ├── model_pruning.py     # 模型剪枝实现
│   └── knowledge_distillation.py  # 知识蒸馏实现
├── quantization/            # 模型量化模块
│   └── model_quantization.py # 量化优化实现
├── caching/                 # 缓存策略模块
│   └── model_cache.py       # 缓存和预加载策略
├── optimization/            # 优化模块
│   ├── deployment_architecture.py  # 部署架构优化
│   └── memory_optimization.py      # 内存优化
├── scripts/                 # 部署脚本
│   ├── optimize_model.py    # 优化主脚本
│   ├── deploy.sh           # 部署脚本
│   └── server.py           # 优化后的Web服务
├── configs/                 # 配置文件
│   └── optimization_config.yaml  # 优化配置
├── docs/                    # 文档
│   └── deployment_guide.md  # 本部署指南
├── Dockerfile              # Docker配置
└── requirements.txt        # 依赖包列表
```

## 功能特性

### 1. 模型压缩

#### 模型剪枝
- **结构化剪枝**: 剪除整个通道或层
- **非结构化剪枝**: 剪除单个权重参数
- **渐进式剪枝**: 逐步增加剪枝比例

#### 知识蒸馏
- **标准蒸馏**: 教师-学生模型知识转移
- **渐进式蒸馏**: 多教师模型渐进蒸馏
- **特征蒸馏**: 中间层特征知识转移

### 2. 模型量化

- **动态量化**: 运行时量化
- **静态量化**: 离线校准量化
- **量化感知训练**: 训练时量化
- **混合精度**: 自适应精度策略

### 3. 缓存和预加载

- **智能缓存**: LRU策略缓存管理
- **模型预加载**: 异步预加载模型
- **自适应缓存**: 基于访问模式的智能缓存
- **内存池管理**: 高效内存分配

### 4. 内存优化

- **智能垃圾回收**: 自动内存清理
- **内存监控**: 实时内存使用监控
- **内存压力预测**: 基于历史数据的预测
- **自适应批处理**: 动态调整批处理大小

### 5. 部署架构

- **本地部署**: 直接部署到本地机器
- **Docker部署**: 容器化部署
- **云部署**: 支持主流云平台
- **系统分析**: 自动分析系统配置并推荐部署方案

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd code/deployment

# 安装依赖
pip install -r requirements.txt

# 给脚本添加执行权限
chmod +x scripts/*.py
chmod +x scripts/deploy.sh
```

### 2. 配置优化

编辑配置文件 `configs/optimization_config.yaml`:

```yaml
# 模型剪枝配置
pruning:
  enabled: true
  type: "unstructured"  # structured, unstructured, progressive
  ratio: 0.5

# 知识蒸馏配置
distillation:
  enabled: false
  type: "standard"
  temperature: 4.0
  alpha: 0.7

# 模型量化配置
quantization:
  enabled: true

# 缓存配置
cache:
  cache_dir: "./model_cache"
  max_cache_size_gb: 2.0

# 监控系统配置
monitoring:
  memory_limit_gb: 4.0
  gc_threshold: 0.8
```

### 3. 运行优化

```bash
# 创建示例配置
python scripts/optimize_model.py --create_config

# 运行完整优化流程
python scripts/optimize_model.py \
    --config configs/optimization_config.yaml \
    --model_path models/best_model.pth \
    --full_pipeline
```

### 4. 部署服务

#### 本地部署

```bash
# 简单部署
./scripts/deploy.sh --deploy-type local --optimize

# 自定义参数部署
./scripts/deploy.sh \
    --deploy-type local \
    --model-path models/optimized_model.pth \
    --port 8080 \
    --workers 4 \
    --batch-size 32 \
    --enable-gpu
```

#### Docker部署

```bash
# 构建并运行Docker容器
./scripts/deploy.sh --deploy-type docker --optimize

# 手动Docker部署
docker build -t traffic-model:latest .
docker run -d \
    --name traffic-model \
    -p 8080:8080 \
    -v $(pwd)/models:/app/models \
    traffic-model:latest
```

### 5. 访问服务

```bash
# 健康检查
curl http://localhost:8080/health

# 预测接口
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"input": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'

# 缓存统计
curl http://localhost:8080/cache/stats

# 资源统计
curl http://localhost:8080/resource/stats
```

## 详细使用说明

### 模型剪枝

```python
from compression.model_pruning import UnstructuredPruner

# 创建剪枝器
pruner = UnstructuredPruner(model, sparsity_ratio=0.5)

# 执行剪枝
pruned_model = pruner.prune_model()

# 微调
pruner.fine_tune(train_loader, epochs=10)
```

### 知识蒸馏

```python
from compression.knowledge_distillation import KnowledgeDistiller

# 创建蒸馏器
distiller = KnowledgeDistiller(
    teacher_model, student_model,
    temperature=4.0, alpha=0.7, beta=0.3
)

# 执行蒸馏
distilled_student = distiller.distill(
    train_loader, val_loader, epochs=50
)
```

### 模型量化

```python
from quantization.model_quantization import QuantizationOptimizer

# 创建量化优化器
optimizer = QuantizationOptimizer(model)

# 运行优化
results = optimizer.optimize_quantization(
    train_loader, val_loader, test_data, test_labels
)

# 获取最佳策略
best_strategy = results['best_strategy']
```

### 缓存管理

```python
from caching.model_cache import ModelCache

# 创建缓存管理器
cache_manager = ModelCache(
    cache_dir='./model_cache',
    max_cache_size_gb=2.0
)

# 缓存模型
cache_key = cache_manager.cache_model(model, "model_id")

# 加载模型
success = cache_manager.load_model(new_model, "model_id")
```

### 内存优化

```python
from optimization.memory_optimization import MemoryOptimizer

# 创建内存优化器
optimizer = MemoryOptimizer(memory_limit_gb=4.0)

# 监控内存
stats = optimizer.monitor_memory()

# 智能垃圾回收
optimizer.smart_gc(force=True)

# 获取优化建议
recommendations = optimizer.get_optimization_recommendations()
```

## 性能优化建议

### 1. 系统要求

#### 最低配置
- CPU: 2核心
- 内存: 4GB
- 存储: 10GB可用空间

#### 推荐配置
- CPU: 8核心
- 内存: 16GB
- GPU: 4GB显存 (可选)
- 存储: 50GB可用空间

#### 云服务器配置
- CPU: 16核心
- 内存: 32GB
- GPU: 8GB显存
- 存储: 100GB SSD

### 2. 优化策略

#### 模型选择
- 对于CPU部署: 优先使用剪枝和量化
- 对于GPU部署: 可使用更大的模型
- 对于边缘设备: 重点使用量化和剪枝

#### 缓存策略
- 热点模型: 启用预加载
- 冷门模型: 延迟加载
- 内存不足: 减少缓存大小

#### 批处理优化
- 高并发: 使用较小的批处理大小
- 低延迟: 优先响应速度
- 高吞吐量: 优化批处理大小

### 3. 监控指标

#### 性能指标
- 推理延迟 (毫秒)
- 吞吐量 (请求/秒)
- 资源使用率 (CPU/内存/GPU)

#### 质量指标
- 模型精度
- 模型大小
- 压缩比

## 故障排除

### 常见问题

#### 1. 内存不足
```
错误: CUDA out of memory
解决: 减少批处理大小，启用梯度累积
```

#### 2. 模型加载失败
```
错误: Model file not found
解决: 检查模型路径，确保文件存在
```

#### 3. 端口占用
```
错误: Port already in use
解决: 更换端口或停止占用进程
```

#### 4. GPU不可用
```
错误: CUDA device not available
解决: 检查GPU驱动，安装CUDA
```

### 调试模式

```bash
# 启用调试日志
export LOG_LEVEL=DEBUG

# 启用详细输出
python scripts/server.py --debug

# 监控资源使用
watch -n 1 'ps aux | grep python'
```

## API文档

### 健康检查接口

```http
GET /health
```

响应:
```json
{
  "status": "healthy",
  "timestamp": 1634567890.123,
  "model_loaded": true,
  "memory_usage": {...},
  "resource_usage": {...}
}
```

### 预测接口

```http
POST /predict
Content-Type: application/json

{
  "input": [[1.0, 2.0, 3.0, 4.0, 5.0]]
}
```

响应:
```json
{
  "prediction": [[0.123, 0.456, 0.789]],
  "inference_time": 0.023,
  "batch_size_used": 32,
  "memory_usage": 65.2
}
```

### 缓存统计接口

```http
GET /cache/stats
```

响应:
```json
{
  "hits": 150,
  "misses": 25,
  "hit_rate": 0.857,
  "cached_models": 5,
  "cache_size_gb": 1.2
}
```

### 资源统计接口

```http
GET /resource/stats
```

响应:
```json
{
  "memory": {
    "total_memory_mb": 8192,
    "used_memory_mb": 4096,
    "memory_percent": 50.0
  },
  "resource": {
    "cpu_percent": 25.5,
    "memory_percent": 50.0,
    "gpu_percent": 0.0
  },
  "batch_size": 32
}
```

## 高级配置

### Kubernetes部署

创建 `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: traffic-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: traffic-model
  template:
    metadata:
      labels:
        app: traffic-model
    spec:
      containers:
      - name: traffic-model
        image: traffic-model:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MODEL_PATH
          value: "/app/models/best_model.pth"
        - name: BATCH_SIZE
          value: "32"
```

### 负载均衡配置

使用Nginx进行负载均衡:

```nginx
upstream traffic_model {
    server localhost:8080;
    server localhost:8081;
    server localhost:8082;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://traffic_model;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 监控集成

#### Prometheus指标

```python
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
REQUEST_COUNT = Counter('model_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('model_request_duration_seconds', 'Request latency')
MEMORY_USAGE = Gauge('model_memory_usage_mb', 'Memory usage in MB')

# 在请求处理中使用
@self.app.before_request
def before_request():
    REQUEST_COUNT.inc()

@self.app.after_request
def after_request(response):
    REQUEST_LATENCY.observe(time.time() - start_time)
    MEMORY_USAGE.set(self.get_memory_stats().used_memory_mb)
    return response
```

## 最佳实践

### 1. 开发阶段
- 使用较小的模型进行开发测试
- 启用详细日志记录
- 定期检查内存使用情况

### 2. 测试阶段
- 进行压力测试
- 验证模型精度
- 检查资源使用效率

### 3. 生产阶段
- 启用缓存策略
- 配置监控告警
- 定期更新模型

### 4. 维护阶段
- 定期清理缓存
- 监控性能指标
- 备份重要配置

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系:

- 邮箱: support@example.com
- GitHub Issues: [项目地址]/issues
- 文档: [文档地址]

---

*最后更新: 2025-11-05*