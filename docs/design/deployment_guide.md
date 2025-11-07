# 交通流预测系统部署和运维指南

## 1. 系统部署架构

### 1.1 硬件要求

#### 最小配置（开发测试环境）
- CPU: 8核 2.5GHz以上
- 内存: 32GB DDR4
- 存储: 500GB SSD + 2TB HDD
- 网络: 1Gbps以太网

#### 推荐配置（生产环境）
- CPU: 32核 3.0GHz以上（Intel Xeon或AMD EPYC）
- 内存: 128GB DDR4 ECC
- 存储: 2TB NVMe SSD + 10TB HDD
- 网络: 10Gbps以太网
- GPU: NVIDIA Tesla V100或A100（可选，用于模型训练）

#### 高可用配置（企业级部署）
- 负载均衡器: 2台HA配置
- 应用服务器: 4台集群部署
- 数据库服务器: 2台主从配置
- 存储服务器: 4台分布式存储集群
- 监控服务器: 1台独立部署

### 1.2 软件环境

#### 基础软件栈
```bash
# 操作系统
Ubuntu 20.04 LTS / CentOS 8

# 运行时环境
Python 3.8+
Node.js 16+
Java 11+

# 数据库
PostgreSQL 13+
Redis 6+
InfluxDB 2.0+

# 流处理
Apache Kafka 3.0+
Apache Storm 2.4+

# 容器化
Docker 20.10+
Kubernetes 1.22+
```

#### 核心依赖库
```python
# 深度学习框架
torch==1.12.0
tensorflow==2.8.0
scikit-learn==1.1.0

# 数据处理
pandas==1.4.0
numpy==1.21.0
geopandas==0.11.0

# 流处理
kafka-python==2.0.2
pystorm==0.6.0

# 可视化
plotly==5.8.0
folium==0.12.0
dash==2.6.0

# 监控
prometheus-client==0.14.0
grafana-api==1.0.3
```

### 1.3 部署步骤

#### 阶段1：环境准备
```bash
# 1. 安装基础软件
sudo apt update
sudo apt install -y python3.8 python3.8-venv docker.io docker-compose

# 2. 创建应用目录
sudo mkdir -p /opt/traffic-prediction/{app,data,logs,config}
sudo chown -R $USER:$USER /opt/traffic-prediction

# 3. 克隆代码仓库
git clone https://github.com/company/traffic-prediction.git
cd traffic-prediction

# 4. 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 阶段2：配置文件设置
```bash
# 1. 复制配置模板
cp config/config.template.yaml config/config.yaml

# 2. 编辑配置文件
vim config/config.yaml

# 主要配置项：
# - 数据库连接信息
# - Kafka集群配置
# - Redis集群配置
# - API密钥配置
# - 模型路径配置
```

#### 阶段3：数据库初始化
```bash
# 1. 启动数据库服务
docker-compose up -d postgres redis influxdb

# 2. 初始化数据库结构
python scripts/init_database.py

# 3. 导入历史数据（可选）
python scripts/import_historical_data.py --data-path /path/to/historical/data
```

#### 阶段4：模型部署
```bash
# 1. 部署预训练模型
python scripts/deploy_model.py --model-path models/gcn_lstm_v1.0.pkl

# 2. 验证模型部署
python scripts/test_model_deployment.py

# 3. 设置模型版本管理
python scripts/setup_model_versioning.py
```

#### 阶段5：服务启动
```bash
# 1. 启动数据接入服务
python -m services.data_ingestion

# 2. 启动流处理服务
python -m services.stream_processing

# 3. 启动模型推理服务
python -m services.model_inference

# 4. 启动Web服务
python -m services.web_interface

# 5. 启动监控服务
python -m services.monitoring
```

## 2. 监控和运维

### 2.1 系统监控

#### 关键监控指标
```yaml
# 监控指标配置
system_metrics:
  cpu_usage:
    warning: 80%
    critical: 95%
  
  memory_usage:
    warning: 85%
    critical: 95%
  
  disk_usage:
    warning: 80%
    critical: 90%
  
  network_latency:
    warning: 100ms
    critical: 500ms

data_metrics:
  data_quality:
    missing_rate_threshold: 5%
    anomaly_rate_threshold: 2%
  
  processing_latency:
    warning: 30s
    critical: 60s
  
  throughput:
    min_events_per_second: 1000
    max_events_per_second: 10000

model_metrics:
  prediction_accuracy:
    min_mae_threshold: 2.0
    min_rmse_threshold: 3.0
  
  model_latency:
    warning: 100ms
    critical: 500ms
  
  model_updates:
    frequency: daily
    auto_retrain: true
```

#### 监控告警配置
```python
# 告警规则配置
alert_rules = {
    'system_overload': {
        'condition': 'cpu_usage > 90% OR memory_usage > 90%',
        'severity': 'critical',
        'notification': ['email', 'sms', 'slack']
    },
    'data_quality_issue': {
        'condition': 'missing_rate > 5% OR anomaly_rate > 2%',
        'severity': 'warning',
        'notification': ['email', 'slack']
    },
    'model_performance_degradation': {
        'condition': 'mae > 3.0 OR rmse > 4.0',
        'severity': 'warning',
        'notification': ['email', 'slack']
    },
    'service_unavailable': {
        'condition': 'service_status == down',
        'severity': 'critical',
        'notification': ['email', 'sms', 'slack', 'phone']
    }
}
```

### 2.2 日志管理

#### 日志配置
```python
# 日志配置
import logging
import logging.handlers
from datetime import datetime

# 创建日志目录
LOG_DIR = '/opt/traffic-prediction/logs'

# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 文件轮转配置
def setup_logging():
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)
    
    # 文件处理器（按天轮转）
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=f'{LOG_DIR}/traffic_prediction.log',
        when='midnight',
        interval=1,
        backupCount=30
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)
    
    # 错误日志处理器
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=f'{LOG_DIR}/error.log',
        when='midnight',
        interval=1,
        backupCount=90
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(error_handler)
    
    return root_logger

# 使用示例
logger = setup_logging()
logger.info("交通流预测系统启动")
logger.error("数据处理错误", exc_info=True)
```

#### 日志分析脚本
```python
#!/usr/bin/env python3
"""
日志分析脚本
分析系统日志，提取关键信息和异常模式
"""

import re
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class LogAnalyzer:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.error_patterns = [
            r'ERROR',
            r'Exception',
            r'Traceback',
            r'Failed to',
            r'Connection timeout',
            r'Memory error'
        ]
    
    def parse_logs(self, start_time=None, end_time=None):
        """解析日志文件"""
        logs = []
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 解析日志行
                log_entry = self.parse_log_line(line)
                if log_entry:
                    # 时间过滤
                    if start_time and log_entry['timestamp'] < start_time:
                        continue
                    if end_time and log_entry['timestamp'] > end_time:
                        continue
                    logs.append(log_entry)
        
        return pd.DataFrame(logs)
    
    def parse_log_line(self, line):
        """解析单行日志"""
        # 匹配标准日志格式
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)'
        match = re.match(pattern, line.strip())
        
        if match:
            timestamp_str, logger_name, level, message = match.groups()
            return {
                'timestamp': datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f'),
                'logger': logger_name,
                'level': level,
                'message': message
            }
        return None
    
    def detect_anomalies(self, logs_df):
        """检测异常模式"""
        anomalies = []
        
        for _, row in logs_df.iterrows():
            # 检查错误模式
            for pattern in self.error_patterns:
                if re.search(pattern, row['message'], re.IGNORECASE):
                    anomalies.append(row)
                    break
        
        return pd.DataFrame(anomalies)
    
    def generate_report(self, logs_df, anomalies_df):
        """生成日志分析报告"""
        report = {
            'total_logs': len(logs_df),
            'error_count': len(logs_df[logs_df['level'] == 'ERROR']),
            'warning_count': len(logs_df[logs_df['level'] == 'WARNING']),
            'anomaly_count': len(anomalies_df),
            'top_error_messages': logs_df[logs_df['level'] == 'ERROR']['message'].value_counts().head(10).to_dict(),
            'hourly_distribution': logs_df.groupby(logs_df['timestamp'].dt.hour).size().to_dict()
        }
        
        return report

# 使用示例
if __name__ == "__main__":
    analyzer = LogAnalyzer('/opt/traffic-prediction/logs/traffic_prediction.log')
    
    # 分析过去24小时的日志
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    logs = analyzer.parse_logs(start_time, end_time)
    anomalies = analyzer.detect_anomalies(logs)
    report = analyzer.generate_report(logs, anomalies)
    
    print("日志分析报告:")
    print(f"总日志数: {report['total_logs']}")
    print(f"错误数: {report['error_count']}")
    print(f"警告数: {report['warning_count']}")
    print(f"异常数: {report['anomaly_count']}")
```

### 2.3 性能优化

#### 系统性能调优
```bash
# 1. 内核参数优化
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf
sysctl -p

# 2. 文件描述符限制
echo '* soft nofile 65536' >> /etc/security/limits.conf
echo '* hard nofile 65536' >> /etc/security/limits.conf

# 3. 内存管理
echo 'vm.swappiness = 10' >> /etc/sysctl.conf
echo 'vm.dirty_ratio = 15' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio = 5' >> /etc/sysctl.conf
```

#### 数据库优化
```sql
-- PostgreSQL优化配置
-- 在postgresql.conf中设置

-- 内存配置
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 1GB

-- 连接配置
max_connections = 200
max_prepared_transactions = 200

-- WAL配置
wal_buffers = 64MB
checkpoint_completion_target = 0.9

-- 查询优化
random_page_cost = 1.1
effective_io_concurrency = 200
```

#### 应用层优化
```python
# 缓存配置优化
CACHE_CONFIG = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            },
            'COMPRESSOR': 'django.utils.compress.lz4.LZ4',
        }
    }
}

# 连接池配置
DATABASE_POOL_CONFIG = {
    'ENGINE': 'django.db.backends.postgresql',
    'NAME': 'traffic_prediction',
    'USER': 'traffic_user',
    'PASSWORD': 'password',
    'HOST': 'localhost',
    'PORT': '5432',
    'OPTIONS': {
        'MAX_CONNS': 20,
        'MIN_CONNS': 5,
    },
}

# 异步处理配置
CELERY_CONFIG = {
    'broker_url': 'redis://localhost:6379/0',
    'result_backend': 'redis://localhost:6379/0',
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'Asia/Shanghai',
    'enable_utc': True,
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    'worker_max_tasks_per_child': 1000,
}
```

## 3. 故障处理和恢复

### 3.1 常见故障类型

#### 数据源故障
```python
# 数据源故障处理
class DataSourceFailureHandler:
    def __init__(self):
        self.fallback_sources = {
            'primary': 'https://api.traffic.gov.cn/data',
            'backup1': 'https://backup.traffic.gov.cn/data',
            'backup2': 'https://emergency.traffic.gov.cn/data'
        }
        self.current_source = 'primary'
        self.failure_count = 0
        self.max_failures = 3
    
    def handle_failure(self, source, error):
        """处理数据源故障"""
        self.failure_count += 1
        
        if self.failure_count >= self.max_failures:
            self.switch_to_backup(source)
        
        # 记录故障
        logger.error(f"数据源故障: {source}, 错误: {error}")
        
        # 触发告警
        self.send_alert(source, error)
    
    def switch_to_backup(self, failed_source):
        """切换到备用数据源"""
        available_backups = [k for k in self.fallback_sources.keys() 
                           if k != failed_source and k != self.current_source]
        
        if available_backups:
            self.current_source = available_backups[0]
            logger.info(f"切换到备用数据源: {self.current_source}")
    
    def send_alert(self, source, error):
        """发送告警"""
        alert_message = {
            'type': 'data_source_failure',
            'source': source,
            'error': str(error),
            'timestamp': datetime.now().isoformat(),
            'severity': 'critical' if self.failure_count >= self.max_failures else 'warning'
        }
        
        # 发送到告警系统
        self.notification_service.send(alert_message)
```

#### 模型推理故障
```python
# 模型推理故障处理
class ModelInferenceFailureHandler:
    def __init__(self):
        self.model_versions = ['v1.0', 'v0.9', 'v0.8']
        self.current_version = 'v1.0'
        self.fallback_models = {}
    
    def handle_inference_failure(self, model_version, error):
        """处理模型推理故障"""
        logger.error(f"模型推理故障: {model_version}, 错误: {error}")
        
        # 尝试回退到稳定版本
        stable_version = self.get_stable_version(model_version)
        if stable_version != model_version:
            self.switch_model(stable_version)
        
        # 记录故障并触发模型重载
        self.schedule_model_reload(model_version)
    
    def get_stable_version(self, current_version):
        """获取稳定版本"""
        if current_version == 'v1.0':
            return 'v0.9'  # 回退到上一个稳定版本
        elif current_version == 'v0.9':
            return 'v0.8'
        else:
            return 'v0.8'
    
    def switch_model(self, version):
        """切换模型版本"""
        try:
            # 加载新模型
            new_model = self.load_model(version)
            
            # 原子性切换
            with self.model_lock:
                self.current_model = new_model
                self.current_version = version
            
            logger.info(f"模型切换成功: {version}")
            
        except Exception as e:
            logger.error(f"模型切换失败: {e}")
            raise
    
    def schedule_model_reload(self, failed_version):
        """安排模型重载"""
        # 使用Celery异步任务重载模型
        from celery import current_app
        current_app.send_task('models.reload_model', args=[failed_version])
```

#### 存储系统故障
```python
# 存储系统故障处理
class StorageFailureHandler:
    def __init__(self):
        self.storage_backends = {
            'primary': 'postgresql://localhost:5432/traffic',
            'backup1': 'postgresql://backup1:5432/traffic',
            'backup2': 'postgresql://backup2:5432/traffic'
        }
        self.current_backend = 'primary'
    
    def handle_storage_failure(self, backend, error):
        """处理存储故障"""
        logger.error(f"存储故障: {backend}, 错误: {error}")
        
        # 切换到备用存储
        if backend == self.current_backend:
            self.switch_storage_backend()
        
        # 启用本地缓存模式
        self.enable_local_cache_mode()
        
        # 触发数据同步
        self.schedule_data_sync(backend)
    
    def switch_storage_backend(self):
        """切换存储后端"""
        available_backends = [k for k in self.storage_backends.keys() 
                            if k != self.current_backend]
        
        if available_backends:
            new_backend = available_backends[0]
            try:
                # 测试新连接
                self.test_connection(new_backend)
                
                # 切换后端
                self.current_backend = new_backend
                logger.info(f"存储后端切换成功: {new_backend}")
                
            except Exception as e:
                logger.error(f"存储后端切换失败: {e}")
                # 继续尝试下一个备用后端
                if len(available_backends) > 1:
                    self.switch_storage_backend()
    
    def enable_local_cache_mode(self):
        """启用本地缓存模式"""
        # 启用本地文件系统缓存
        self.cache_backend = 'file:///tmp/traffic_cache'
        
        # 调整缓存策略
        self.cache_ttl = 3600  # 1小时
        self.cache_max_size = '10GB'
        
        logger.info("已启用本地缓存模式")
```

### 3.2 恢复流程

#### 自动恢复机制
```python
# 自动恢复管理器
class AutoRecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            'data_source': DataSourceFailureHandler(),
            'model_inference': ModelInferenceFailureHandler(),
            'storage': StorageFailureHandler()
        }
        self.health_check_interval = 30  # 秒
        self.max_recovery_attempts = 3
    
    def start_monitoring(self):
        """启动健康监控"""
        import threading
        import time
        
        def health_check_loop():
            while True:
                try:
                    self.perform_health_checks()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"健康检查异常: {e}")
        
        monitor_thread = threading.Thread(target=health_check_loop, daemon=True)
        monitor_thread.start()
    
    def perform_health_checks(self):
        """执行健康检查"""
        # 检查数据源
        if not self.check_data_source_health():
            self.trigger_recovery('data_source')
        
        # 检查模型推理
        if not self.check_model_health():
            self.trigger_recovery('model_inference')
        
        # 检查存储系统
        if not self.check_storage_health():
            self.trigger_recovery('storage')
    
    def check_data_source_health(self):
        """检查数据源健康状态"""
        try:
            response = requests.get('https://api.traffic.gov.cn/health', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_model_health(self):
        """检查模型健康状态"""
        try:
            # 发送测试请求到模型服务
            test_data = self.get_test_data()
            result = self.model_service.predict(test_data)
            return result is not None
        except:
            return False
    
    def check_storage_health(self):
        """检查存储健康状态"""
        try:
            # 测试数据库连接
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except:
            return False
    
    def trigger_recovery(self, component_type):
        """触发恢复流程"""
        if component_type in self.recovery_strategies:
            handler = self.recovery_strategies[component_type]
            
            for attempt in range(self.max_recovery_attempts):
                try:
                    logger.info(f"尝试恢复 {component_type}, 第 {attempt + 1} 次")
                    
                    if component_type == 'data_source':
                        handler.handle_failure('primary', 'health_check_failed')
                    elif component_type == 'model_inference':
                        handler.handle_inference_failure('v1.0', 'health_check_failed')
                    elif component_type == 'storage':
                        handler.handle_storage_failure('primary', 'health_check_failed')
                    
                    # 等待恢复
                    time.sleep(10)
                    
                    # 验证恢复
                    if self.verify_recovery(component_type):
                        logger.info(f"{component_type} 恢复成功")
                        return True
                        
                except Exception as e:
                    logger.error(f"恢复尝试失败: {e}")
            
            logger.error(f"{component_type} 恢复失败，已达到最大尝试次数")
            self.send_critical_alert(component_type)
            return False
    
    def verify_recovery(self, component_type):
        """验证恢复结果"""
        if component_type == 'data_source':
            return self.check_data_source_health()
        elif component_type == 'model_inference':
            return self.check_model_health()
        elif component_type == 'storage':
            return self.check_storage_health()
        return False
    
    def send_critical_alert(self, component_type):
        """发送严重告警"""
        alert_message = {
            'type': 'critical_system_failure',
            'component': component_type,
            'message': f'自动恢复失败，需要人工干预',
            'timestamp': datetime.now().isoformat(),
            'severity': 'critical'
        }
        
        self.notification_service.send_critical_alert(alert_message)
```

#### 手动恢复流程
```bash
#!/bin/bash
# 手动恢复脚本

echo "开始系统恢复流程..."

# 1. 检查系统状态
echo "检查系统状态..."
python scripts/system_health_check.py

# 2. 重启服务
echo "重启核心服务..."
docker-compose restart kafka storm redis postgres

# 3. 重新加载配置
echo "重新加载配置..."
python scripts/reload_configuration.py

# 4. 验证数据连接
echo "验证数据连接..."
python scripts/test_data_connections.py

# 5. 验证模型部署
echo "验证模型部署..."
python scripts/verify_model_deployment.py

# 6. 重新启动应用服务
echo "重新启动应用服务..."
python -m services.data_ingestion &
python -m services.stream_processing &
python -m services.model_inference &
python -m services.web_interface &

# 7. 最终验证
echo "最终系统验证..."
python scripts/final_system_check.py

echo "系统恢复完成"
```

## 4. 安全配置

### 4.1 网络安全

#### 防火墙配置
```bash
#!/bin/bash
# 防火墙配置脚本

# 清除现有规则
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X

# 默认策略
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# 允许本地回环
iptables -A INPUT -i lo -j ACCEPT

# 允许已建立的连接
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# 允许SSH（限制IP范围）
iptables -A INPUT -p tcp --dport 22 -s 192.168.1.0/24 -j ACCEPT

# 允许HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# 允许应用端口
iptables -A INPUT -p tcp --dport 8080 -j ACCEPT  # Web界面
iptables -A INPUT -p tcp --dport 9092 -j ACCEPT  # Kafka
iptables -A INPUT -p tcp --dport 5432 -j ACCEPT  # PostgreSQL
iptables -A INPUT -p tcp --dport 6379 -j ACCEPT  # Redis

# 限制连接频率
iptables -A INPUT -p tcp --dport 22 -m limit --limit 3/min --limit-burst 5 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -m limit --limit 10/min --limit-burst 20 -j ACCEPT

# 记录被拒绝的连接
iptables -A INPUT -j LOG --log-prefix "iptables denied: "
iptables -A INPUT -j DROP

# 保存规则
iptables-save > /etc/iptables/rules.v4
```

#### SSL/TLS配置
```nginx
# Nginx SSL配置 (/etc/nginx/sites-available/traffic-prediction)

server {
    listen 443 ssl http2;
    server_name traffic-prediction.company.com;
    
    # SSL证书配置
    ssl_certificate /etc/ssl/certs/traffic-prediction.crt;
    ssl_certificate_key /etc/ssl/private/traffic-prediction.key;
    
    # SSL安全配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # 安全头
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # 代理到应用服务
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时配置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # API接口限流
    location /api/ {
        proxy_pass http://localhost:8080;
        limit_req zone=api burst=20 nodelay;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# HTTP重定向到HTTPS
server {
    listen 80;
    server_name traffic-prediction.company.com;
    return 301 https://$server_name$request_uri;
}
```

### 4.2 身份认证和授权

#### JWT认证配置
```python
# JWT认证配置
import jwt
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

# JWT配置
JWT_SECRET_KEY = 'your-secret-key-here'
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_DELTA = timedelta(hours=24)

class AuthenticationManager:
    def __init__(self):
        self.users = {
            'admin': {
                'password': 'hashed_password_here',
                'role': 'admin',
                'permissions': ['read', 'write', 'admin']
            },
            'operator': {
                'password': 'hashed_password_here',
                'role': 'operator',
                'permissions': ['read', 'write']
            },
            'viewer': {
                'password': 'hashed_password_here',
                'role': 'viewer',
                'permissions': ['read']
            }
        }
    
    def generate_token(self, username):
        """生成JWT令牌"""
        payload = {
            'username': username,
            'role': self.users[username]['role'],
            'permissions': self.users[username]['permissions'],
            'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA,
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    
    def verify_token(self, token):
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def require_auth(self, required_permissions=None):
        """认证装饰器"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                token = request.headers.get('Authorization')
                
                if not token:
                    return jsonify({'error': '缺少认证令牌'}), 401
                
                # 移除Bearer前缀
                if token.startswith('Bearer '):
                    token = token[7:]
                
                payload = self.verify_token(token)
                if not payload:
                    return jsonify({'error': '无效或过期的令牌'}), 401
                
                # 检查权限
                if required_permissions:
                    user_permissions = payload.get('permissions', [])
                    if not any(perm in user_permissions for perm in required_permissions):
                        return jsonify({'error': '权限不足'}), 403
                
                # 将用户信息添加到请求上下文
                request.current_user = payload
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator

# 使用示例
auth_manager = AuthenticationManager()

@app.route('/login', methods=['POST'])
def login():
    """用户登录"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if username in auth_manager.users:
        user = auth_manager.users[username]
        # 在实际应用中，应该使用安全的密码哈希验证
        if password == user['password']:  # 简化示例
            token = auth_manager.generate_token(username)
            return jsonify({
                'token': token,
                'user': {
                    'username': username,
                    'role': user['role'],
                    'permissions': user['permissions']
                }
            })
    
    return jsonify({'error': '用户名或密码错误'}), 401

@app.route('/api/predictions')
@auth_manager.require_auth(['read'])
def get_predictions():
    """获取预测结果（需要读权限）"""
    # 获取预测数据
    predictions = get_latest_predictions()
    return jsonify(predictions)

@app.route('/api/admin/models', methods=['POST'])
@auth_manager.require_auth(['admin'])
def upload_model():
    """上传模型（需要管理员权限）"""
    # 上传模型逻辑
    return jsonify({'message': '模型上传成功'})
```

#### RBAC权限控制
```python
# 基于角色的访问控制 (RBAC)
class RBACManager:
    def __init__(self):
        self.roles = {
            'admin': {
                'permissions': ['read', 'write', 'delete', 'admin', 'system_config'],
                'description': '系统管理员'
            },
            'operator': {
                'permissions': ['read', 'write', 'data_management'],
                'description': '操作员'
            },
            'analyst': {
                'permissions': ['read', 'write', 'data_analysis'],
                'description': '数据分析师'
            },
            'viewer': {
                'permissions': ['read'],
                'description': '只读用户'
            }
        }
        
        self.resources = {
            'predictions': ['read'],
            'models': ['read', 'write', 'delete'],
            'data_sources': ['read', 'write'],
            'users': ['read', 'write', 'delete'],
            'system_config': ['read', 'write'],
            'logs': ['read'],
            'reports': ['read', 'write']
        }
    
    def check_permission(self, user_role, resource, action):
        """检查用户权限"""
        if user_role not in self.roles:
            return False
        
        user_permissions = self.roles[user_role]['permissions']
        
        # 检查资源权限
        if resource in self.resources:
            required_permission = action
            return required_permission in user_permissions
        
        # 检查全局权限
        return action in user_permissions
    
    def require_permission(self, resource, action):
        """权限检查装饰器"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                current_user = getattr(request, 'current_user', None)
                if not current_user:
                    return jsonify({'error': '用户未认证'}), 401
                
                user_role = current_user.get('role')
                if not self.check_permission(user_role, resource, action):
                    return jsonify({'error': f'权限不足：需要{resource}的{action}权限'}), 403
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator

# 使用示例
rbac = RBACManager()

@app.route('/api/models', methods=['POST'])
@auth_manager.require_auth(['read'])
@rbac.require_permission('models', 'write')
def upload_model():
    """上传模型（需要models的write权限）"""
    # 上传模型逻辑
    pass

@app.route('/api/users', methods=['GET'])
@auth_manager.require_auth(['read'])
@rbac.require_permission('users', 'read')
def get_users():
    """获取用户列表（需要users的read权限）"""
    # 获取用户逻辑
    pass
```

### 4.3 数据安全

#### 数据加密
```python
# 数据加密工具
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self, password: bytes):
        # 生成盐值
        self.salt = os.urandom(16)
        
        # 派生密钥
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        encrypted_data = self.cipher.encrypt(data.encode())
        # 将盐值和加密数据合并
        combined = self.salt + encrypted_data
        return base64.urlsafe_b64encode(combined).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            # 解码
            combined = base64.urlsafe_b64decode(encrypted_data.encode())
            
            # 提取盐值和加密数据
            salt = combined[:16]
            encrypted = combined[16:]
            
            # 重新派生密钥（需要相同的密码和盐值）
            # 注意：在实际应用中，盐值应该与加密数据一起存储
            # 这里为了简化示例，假设使用相同的盐值
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(b'your_password'))
            cipher = Fernet(key)
            
            decrypted_data = cipher.decrypt(encrypted)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"解密失败: {e}")

# 数据库字段加密
class EncryptedField:
    def __init__(self, password):
        self.encryptor = DataEncryption(password.encode())
    
    def encrypt_value(self, value):
        """加密字段值"""
        if value is None:
            return None
        return self.encryptor.encrypt(str(value))
    
    def decrypt_value(self, encrypted_value):
        """解密字段值"""
        if encrypted_value is None:
            return None
        return self.encryptor.decrypt(encrypted_value)

# 使用示例
# 敏感数据加密
sensitive_data_encryptor = EncryptedField(b'sensitive_data_password')

# 加密存储
def store_sensitive_data(user_id, personal_info):
    encrypted_data = {
        'user_id': user_id,
        'name': sensitive_data_encryptor.encrypt_value(personal_info['name']),
        'phone': sensitive_data_encryptor.encrypt_value(personal_info['phone']),
        'email': sensitive_data_encryptor.encrypt_value(personal_info['email'])
    }
    
    # 存储到数据库
    db.users.insert_one(encrypted_data)

# 解密读取
def get_sensitive_data(user_id):
    user_data = db.users.find_one({'user_id': user_id})
    if user_data:
        return {
            'user_id': user_data['user_id'],
            'name': sensitive_data_encryptor.decrypt_value(user_data['name']),
            'phone': sensitive_data_encryptor.decrypt_value(user_data['phone']),
            'email': sensitive_data_encryptor.decrypt_value(user_data['email'])
        }
    return None
```

#### 数据脱敏
```python
# 数据脱敏工具
import re
import hashlib

class DataMasking:
    @staticmethod
    def mask_phone_number(phone):
        """脱敏手机号"""
        if not phone or len(phone) != 11:
            return phone
        
        return phone[:3] + '****' + phone[7:]
    
    @staticmethod
    def mask_email(email):
        """脱敏邮箱"""
        if not email or '@' not in email:
            return email
        
        username, domain = email.split('@')
        if len(username) <= 2:
            masked_username = username
        else:
            masked_username = username[:2] + '*' * (len(username) - 2)
        
        return f"{masked_username}@{domain}"
    
    @staticmethod
    def mask_id_number(id_number):
        """脱敏身份证号"""
        if not id_number or len(id_number) != 18:
            return id_number
        
        return id_number[:6] + '*' * 8 + id_number[-4:]
    
    @staticmethod
    def hash_sensitive_data(data):
        """哈希敏感数据"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def mask_location(location):
        """脱敏位置信息"""
        # 只保留到区县级
        if ',' in location:
            parts = location.split(',')
            if len(parts) >= 2:
                return f"{parts[0]},{parts[1]}"
        return location

# 使用示例
masker = DataMasking()

# 日志脱敏
def log_with_masking(user_data):
    masked_data = {
        'user_id': user_data['user_id'],
        'phone': masker.mask_phone_number(user_data.get('phone')),
        'email': masker.mask_email(user_data.get('email')),
        'location': masker.mask_location(user_data.get('location'))
    }
    
    logger.info(f"用户数据: {masked_data}")

# API响应脱敏
@app.route('/api/user/profile')
@auth_manager.require_auth(['read'])
def get_user_profile():
    user = get_current_user()
    
    # 脱敏敏感字段
    profile = {
        'user_id': user['user_id'],
        'name': user['name'],
        'phone': masker.mask_phone_number(user.get('phone')),
        'email': masker.mask_email(user.get('email')),
        'location': masker.mask_location(user.get('location'))
    }
    
    return jsonify(profile)
```

## 5. 备份和灾难恢复

### 5.1 数据备份策略

#### 数据库备份
```bash
#!/bin/bash
# 数据库备份脚本

# 配置
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="traffic_prediction"
DB_USER="traffic_user"
BACKUP_DIR="/opt/traffic-prediction/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 数据库备份
echo "开始数据库备份..."
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
    --format=custom \
    --verbose \
    --file="$BACKUP_DIR/traffic_db_$DATE.dump"

# 压缩备份
gzip "$BACKUP_DIR/traffic_db_$DATE.dump"

# 上传到云存储（可选）
if [ "$1" = "upload" ]; then
    echo "上传备份到云存储..."
    aws s3 cp "$BACKUP_DIR/traffic_db_$DATE.dump.gz" s3://traffic-prediction-backups/database/
fi

# 清理旧备份（保留最近30天）
find $BACKUP_DIR -name "traffic_db_*.dump.gz" -mtime +30 -delete

echo "数据库备份完成: traffic_db_$DATE.dump.gz"
```

#### 文件系统备份
```bash
#!/bin/bash
# 文件系统备份脚本

# 配置
SOURCE_DIRS=(
    "/opt/traffic-prediction"
    "/etc/nginx/sites-available"
    "/etc/ssl/certs"
)

BACKUP_DIR="/opt/traffic-prediction/backups/files"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份配置文件和代码
echo "开始文件系统备份..."
tar -czf "$BACKUP_DIR/filesystem_$DATE.tar.gz" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='node_modules' \
    --exclude='.git' \
    "${SOURCE_DIRS[@]}"

# 备份Docker配置
docker-compose config > "$BACKUP_DIR/docker_compose_$DATE.yml"

# 备份Nginx配置
cp -r /etc/nginx/sites-available "$BACKUP_DIR/nginx_config_$DATE/"

# 上传到云存储（可选）
if [ "$1" = "upload" ]; then
    echo "上传备份到云存储..."
    aws s3 cp "$BACKUP_DIR/filesystem_$DATE.tar.gz" s3://traffic-prediction-backups/filesystem/
fi

# 清理旧备份（保留最近7天）
find $BACKUP_DIR -name "filesystem_*.tar.gz" -mtime +7 -delete

echo "文件系统备份完成: filesystem_$DATE.tar.gz"
```

#### 自动化备份调度
```bash
# 添加到crontab

# 每天凌晨2点备份数据库
0 2 * * * /opt/traffic-prediction/scripts/backup_database.sh upload

# 每周日凌晨3点备份文件系统
0 3 * * 0 /opt/traffic-prediction/scripts/backup_filesystem.sh upload

# 每月1号清理旧备份
0 4 1 * * find /opt/traffic-prediction/backups -name "*.dump.gz" -mtime +90 -delete
```

### 5.2 灾难恢复流程

#### 数据库恢复
```bash
#!/bin/bash
# 数据库恢复脚本

# 配置
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="traffic_prediction"
DB_USER="traffic_user"
BACKUP_DIR="/opt/traffic-prediction/backups"

# 列出可用备份
echo "可用备份文件："
ls -la $BACKUP_DIR/traffic_db_*.dump.gz

# 选择要恢复的备份
read -p "请输入要恢复的备份文件名: " BACKUP_FILE

if [ ! -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
    echo "备份文件不存在: $BACKUP_FILE"
    exit 1
fi

# 停止应用服务
echo "停止应用服务..."
docker-compose stop

# 解压备份文件
echo "解压备份文件..."
gunzip -c "$BACKUP_DIR/$BACKUP_FILE" > "/tmp/traffic_db_$(date +%Y%m%d_%H%M%S).dump"

# 删除现有数据库（谨慎操作）
echo "删除现有数据库..."
dropdb -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME

# 创建新数据库
echo "创建新数据库..."
createdb -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME

# 恢复数据
echo "恢复数据..."
RESTORE_FILE="/tmp/traffic_db_$(date +%Y%m%d_%H%M%S).dump"
pg_restore -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
    --verbose \
    --clean \
    --if-exists \
    $RESTORE_FILE

# 清理临时文件
rm -f $RESTORE_FILE

# 启动应用服务
echo "启动应用服务..."
docker-compose start

echo "数据库恢复完成"
```

#### 系统完整恢复
```bash
#!/bin/bash
# 系统完整恢复脚本

echo "开始系统完整恢复流程..."

# 1. 停止所有服务
echo "停止所有服务..."
docker-compose down

# 2. 恢复文件系统
echo "恢复文件系统..."
read -p "请输入文件系统备份文件名: " BACKUP_FILE
tar -xzf "/opt/traffic-prediction/backups/$BACKUP_FILE" -C /

# 3. 恢复数据库
echo "恢复数据库..."
read -p "请输入数据库备份文件名: " DB_BACKUP
gunzip -c "/opt/traffic-prediction/backups/$DB_BACKUP" > "/tmp/restore_db.dump"
dropdb -h localhost -p 5432 -U traffic_user traffic_prediction
createdb -h localhost -p 5432 -U traffic_user traffic_prediction
pg_restore -h localhost -p 5432 -U traffic_user -d traffic_prediction --verbose --clean /tmp/restore_db.dump
rm -f /tmp/restore_db.dump

# 4. 恢复Docker配置
echo "恢复Docker配置..."
docker-compose -f "/opt/traffic-prediction/backups/docker_compose_$(date +%Y%m%d)*.yml" config > docker-compose.yml

# 5. 重新构建和启动服务
echo "重新构建和启动服务..."
docker-compose build
docker-compose up -d

# 6. 验证恢复
echo "验证系统恢复..."
sleep 30
python scripts/system_health_check.py

echo "系统恢复完成"
```

### 5.3 业务连续性计划

#### 应急响应流程
```python
# 应急响应管理器
class EmergencyResponseManager:
    def __init__(self):
        self.incident_levels = {
            'low': {'response_time': '1小时', 'escalation': False},
            'medium': {'response_time': '30分钟', 'escalation': True},
            'high': {'response_time': '15分钟', 'escalation': True},
            'critical': {'response_time': '5分钟', 'escalation': True}
        }
        
        self.notification_contacts = {
            'technical_lead': '+86-138-xxxx-xxxx',
            'system_admin': '+86-139-xxxx-xxxx',
            'business_owner': '+86-137-xxxx-xxxx',
            'emergency_team': ['team@company.com']
        }
    
    def handle_incident(self, incident_type, severity, description):
        """处理突发事件"""
        incident_id = self.generate_incident_id()
        level_config = self.incident_levels.get(severity, self.incident_levels['medium'])
        
        # 记录事件
        incident_record = {
            'incident_id': incident_id,
            'type': incident_type,
            'severity': severity,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'status': 'open',
            'response_time': level_config['response_time']
        }
        
        # 立即通知相关人员
        self.send_notifications(incident_record)
        
        # 执行自动恢复
        self.execute_recovery_procedures(incident_type)
        
        # 启动应急流程
        self.start_emergency_procedures(incident_record)
        
        return incident_id
    
    def generate_incident_id(self):
        """生成事件ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = str(hash(timestamp))[-4:]
        return f"INC_{timestamp}_{random_suffix}"
    
    def send_notifications(self, incident_record):
        """发送通知"""
        severity = incident_record['severity']
        
        # 根据严重级别选择通知方式
        if severity in ['high', 'critical']:
            # 严重级别：电话 + 邮件 + 短信
            self.send_sms_notification(incident_record)
            self.send_email_notification(incident_record)
            self.make_emergency_calls(incident_record)
        elif severity == 'medium':
            # 中等级别：邮件 + 短信
            self.send_email_notification(incident_record)
            self.send_sms_notification(incident_record)
        else:
            # 低级别：仅邮件
            self.send_email_notification(incident_record)
    
    def execute_recovery_procedures(self, incident_type):
        """执行恢复程序"""
        recovery_procedures = {
            'data_source_failure': self.recover_data_source,
            'model_failure': self.recover_model,
            'storage_failure': self.recover_storage,
            'network_failure': self.recover_network,
            'security_breach': self.handle_security_breach
        }
        
        if incident_type in recovery_procedures:
            try:
                recovery_procedures[incident_type]()
            except Exception as e:
                logger.error(f"恢复程序执行失败: {e}")
    
    def start_emergency_procedures(self, incident_record):
        """启动应急流程"""
        # 激活备用系统
        if incident_record['severity'] == 'critical':
            self.activate_backup_systems()
        
        # 通知业务方
        self.notify_business_stakeholders(incident_record)
        
        # 启动事件响应团队
        self.activate_incident_response_team(incident_record)
    
    def activate_backup_systems(self):
        """激活备用系统"""
        logger.info("激活备用系统...")
        
        # 切换到备用数据中心
        self.switch_to_backup_datacenter()
        
        # 启用降级服务模式
        self.enable_degraded_service_mode()
        
        # 启用缓存模式
        self.enable_cache_only_mode()
    
    def switch_to_backup_datacenter(self):
        """切换到备用数据中心"""
        try:
            # 更新DNS记录
            self.update_dns_records('backup-datacenter.company.com')
            
            # 切换负载均衡器配置
            self.update_load_balancer_config('backup')
            
            logger.info("已切换到备用数据中心")
        except Exception as e:
            logger.error(f"切换数据中心失败: {e}")
    
    def enable_degraded_service_mode(self):
        """启用降级服务模式"""
        # 禁用非核心功能
        self.disable_non_critical_features()
        
        # 降低数据处理频率
        self.reduce_processing_frequency()
        
        # 启用简化预测模型
        self.switch_to_simple_prediction_model()
    
    def enable_cache_only_mode(self):
        """启用仅缓存模式"""
        # 停止实时数据处理
        self.stop_realtime_processing()
        
        # 启用静态数据缓存
        self.enable_static_data_cache()
        
        # 提供历史预测结果
        self.serve_historical_predictions()
```

---

## 结论

本交通流预测系统数据流和处理流程设计文档提供了一个完整的解决方案，涵盖了从数据输入到结果输出的全流程。系统采用现代化的技术架构，具备以下特点：

### 核心优势
1. **高可用性**：通过分布式架构、负载均衡和故障转移机制确保系统稳定运行
2. **实时性**：采用流处理技术实现秒级数据处理和预测结果输出
3. **可扩展性**：微服务架构支持水平扩展，适应业务增长需求
4. **智能化**：集成GCN+LSTM混合模型，提供准确的交通流预测
5. **可观测性**：完善的监控和告警系统，确保系统健康运行

### 技术特色
1. **多源数据融合**：整合交通、天气、事故等多维数据
2. **动态图建模**：实时更新路网拓扑关系
3. **时空特征工程**：构建丰富的时空特征提升预测精度
4. **智能告警机制**：基于预测结果的主动告警系统
5. **可视化展示**：直观的实时监控和预测结果展示

### 运维保障
1. **完善的监控体系**：多层次、全方位的系统监控
2. **自动化运维**：自动故障检测、恢复和处理
3. **安全防护**：多层次的安全防护机制
4. **灾难恢复**：完整的数据备份和恢复策略
5. **应急响应**：快速的事件响应和恢复流程

该系统设计充分考虑了实际部署和运维需求，提供了从开发测试到生产部署的完整解决方案，为城市交通管理提供了强有力的技术支撑。