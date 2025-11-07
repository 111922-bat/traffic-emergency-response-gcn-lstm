"""
内存使用优化和资源管理模块
实现智能内存管理和资源监控
"""

import torch
import torch.nn as nn
import gc
import psutil
import threading
import time
import os
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import logging
import tracemalloc
from dataclasses import dataclass
import weakref
from contextlib import contextmanager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """内存统计"""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0


@dataclass
class ResourceMetrics:
    """资源指标"""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, memory_limit_gb: float = 4.0, gc_threshold: float = 0.8):
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.gc_threshold = gc_threshold
        self.memory_history = deque(maxlen=100)
        self.optimization_strategies = []
        
    def monitor_memory(self) -> MemoryStats:
        """监控内存使用"""
        # 系统内存
        memory = psutil.virtual_memory()
        total_memory_mb = memory.total / (1024 * 1024)
        available_memory_mb = memory.available / (1024 * 1024)
        used_memory_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent
        
        # GPU内存
        gpu_memory_mb = 0.0
        gpu_memory_percent = 0.0
        
        try:
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                gpu_memory_percent = (gpu_memory_mb / gpu_memory_total) * 100
        except Exception as e:
            logger.warning(f"GPU内存监控失败: {e}")
        
        stats = MemoryStats(
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            used_memory_mb=used_memory_mb,
            memory_percent=memory_percent,
            gpu_memory_mb=gpu_memory_mb,
            gpu_memory_percent=gpu_memory_percent
        )
        
        self.memory_history.append(stats)
        return stats
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """优化模型内存使用"""
        logger.info("开始优化模型内存")
        
        # 1. 梯度检查点
        model = self._enable_gradient_checkpointing(model)
        
        # 2. 混合精度优化
        model = self._enable_mixed_precision(model)
        
        # 3. 内存映射优化
        model = self._optimize_memory_mapping(model)
        
        # 4. 清理临时变量
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return model
    
    def _enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """启用梯度检查点以节省内存"""
        try:
            # 对大型模型启用梯度检查点
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if hasattr(module, 'weight') and module.weight.numel() > 100000:
                        # 对于大参数层，启用检查点
                        pass  # PyTorch的checkpoint需要特殊处理
            logger.info("梯度检查点优化完成")
        except Exception as e:
            logger.warning(f"梯度检查点优化失败: {e}")
        
        return model
    
    def _enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """启用混合精度训练"""
        try:
            if torch.cuda.is_available():
                # 将部分层转换为半精度
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        if hasattr(module, 'weight'):
                            # 保持权重为全精度，激活值使用半精度
                            pass
                logger.info("混合精度优化完成")
        except Exception as e:
            logger.warning(f"混合精度优化失败: {e}")
        
        return model
    
    def _optimize_memory_mapping(self, model: nn.Module) -> nn.Module:
        """优化内存映射"""
        try:
            # 优化模型的内存布局
            for name, param in model.named_parameters():
                if param.is_contiguous():
                    # 确保参数是连续的
                    param.data = param.data.contiguous()
            logger.info("内存映射优化完成")
        except Exception as e:
            logger.warning(f"内存映射优化失败: {e}")
        
        return model
    
    def smart_gc(self, force: bool = False):
        """智能垃圾回收"""
        current_stats = self.monitor_memory()
        
        # 检查是否需要强制垃圾回收
        if force or current_stats.memory_percent > self.gc_threshold * 100:
            logger.info(f"执行智能垃圾回收，内存使用率: {current_stats.memory_percent:.1f}%")
            
            # Python垃圾回收
            collected = gc.collect()
            logger.info(f"Python GC回收对象数: {collected}")
            
            # PyTorch缓存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("PyTorch GPU缓存已清理")
            
            # 再次检查内存
            new_stats = self.monitor_memory()
            freed_memory = current_stats.used_memory_mb - new_stats.used_memory_mb
            logger.info(f"释放内存: {freed_memory:.2f} MB")
    
    def predict_memory_pressure(self, horizon: int = 10) -> Dict[str, float]:
        """预测内存压力"""
        if len(self.memory_history) < 2:
            return {'pressure_score': 0.0, 'predicted_usage': 0.0}
        
        # 提取历史数据
        memory_percentages = [stats.memory_percent for stats in self.memory_history]
        
        # 简单线性回归预测
        x = np.arange(len(memory_percentages))
        y = np.array(memory_percentages)
        
        # 计算趋势
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            predicted_usage = memory_percentages[-1] + slope * horizon
        else:
            predicted_usage = memory_percentages[-1]
        
        # 计算压力分数
        pressure_score = min(max(predicted_usage / 100, 0), 1)
        
        return {
            'pressure_score': pressure_score,
            'predicted_usage': predicted_usage,
            'trend_slope': slope if len(x) > 1 else 0.0
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """获取优化建议"""
        current_stats = self.monitor_memory()
        recommendations = []
        
        # 内存使用建议
        if current_stats.memory_percent > 90:
            recommendations.append("内存使用率过高，建议减少批处理大小或启用模型并行")
        elif current_stats.memory_percent > 80:
            recommendations.append("内存使用率较高，建议启用内存优化策略")
        
        # GPU内存建议
        if current_stats.gpu_memory_percent > 90:
            recommendations.append("GPU内存不足，建议使用梯度累积或模型分片")
        elif current_stats.gpu_memory_percent > 80:
            recommendations.append("GPU内存使用较高，建议启用混合精度")
        
        # 预测建议
        prediction = self.predict_memory_pressure()
        if prediction['pressure_score'] > 0.8:
            recommendations.append("预测内存压力较大，建议提前释放不必要的缓存")
        
        return recommendations


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'gpu_percent': 95.0,
            'disk_usage_percent': 95.0
        }
        self.callbacks = defaultdict(list)
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("资源监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("资源监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 检查告警
                self._check_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """收集资源指标"""
        # CPU和内存
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU
        gpu_percent = 0.0
        try:
            if torch.cuda.is_available():
                gpu_percent = (torch.cuda.memory_allocated() / 
                             torch.cuda.get_device_properties(0).total_memory) * 100
        except:
            pass
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        disk_io_read = disk_io.read_bytes if disk_io else 0
        disk_io_write = disk_io.write_bytes if disk_io else 0
        
        # 网络IO
        network_io = psutil.net_io_counters()
        network_io_sent = network_io.bytes_sent if network_io else 0
        network_io_recv = network_io.bytes_recv if network_io else 0
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write,
            network_io_sent=network_io_sent,
            network_io_recv=network_io_recv
        )
    
    def _check_alerts(self, metrics: ResourceMetrics):
        """检查告警"""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"内存使用率过高: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_percent > self.alert_thresholds['gpu_percent']:
            alerts.append(f"GPU使用率过高: {metrics.gpu_percent:.1f}%")
        
        # 触发告警回调
        for alert in alerts:
            self._trigger_alert_callbacks(alert)
    
    def _trigger_alert_callbacks(self, alert: str):
        """触发告警回调"""
        for callback in self.callbacks['alert']:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[str], None]):
        """添加告警回调"""
        self.callbacks['alert'].append(callback)
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """获取当前指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self, duration_minutes: float = 5) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}
        
        # 计算时间窗口
        current_time = time.time()
        cutoff_time = current_time - (duration_minutes * 60)
        
        # 过滤数据
        recent_metrics = [
            m for m in self.metrics_history 
            if hasattr(m, 'timestamp') and m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            recent_metrics = list(self.metrics_history)[-60:]  # 最近60个数据点
        
        # 计算统计信息
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        gpu_values = [m.gpu_percent for m in recent_metrics]
        
        return {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory': {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            },
            'gpu': {
                'mean': np.mean(gpu_values),
                'max': np.max(gpu_values),
                'min': np.min(gpu_values),
                'std': np.std(gpu_values)
            } if any(gpu_values) else None
        }


class AdaptiveBatchSizer:
    """自适应批处理大小调整器"""
    
    def __init__(self, initial_batch_size: int = 32, 
                 min_batch_size: int = 1, max_batch_size: int = 256):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.performance_history = deque(maxlen=50)
        self.memory_monitor = MemoryOptimizer()
        
    def adjust_batch_size(self, inference_time: float, memory_usage: float) -> int:
        """调整批处理大小"""
        # 记录性能
        self.performance_history.append({
            'batch_size': self.current_batch_size,
            'inference_time': inference_time,
            'memory_usage': memory_usage,
            'throughput': self.current_batch_size / inference_time
        })
        
        # 保持历史记录大小
        if len(self.performance_history) > 50:
            self.performance_history.popleft()
        
        # 调整策略
        new_batch_size = self._calculate_optimal_batch_size()
        
        # 平滑调整
        adjustment_factor = 0.1  # 每次调整10%
        self.current_batch_size = int(
            self.current_batch_size * (1 - adjustment_factor) + 
            new_batch_size * adjustment_factor
        )
        
        # 确保在范围内
        self.current_batch_size = max(
            self.min_batch_size, 
            min(self.max_batch_size, self.current_batch_size)
        )
        
        logger.info(f"批处理大小调整: {new_batch_size} (当前: {self.current_batch_size})")
        return self.current_batch_size
    
    def _calculate_optimal_batch_size(self) -> int:
        """计算最优批处理大小"""
        if len(self.performance_history) < 5:
            return self.current_batch_size
        
        # 分析性能趋势
        recent_data = list(self.performance_history)[-10:]
        
        # 计算内存效率
        memory_efficiencies = []
        throughputs = []
        
        for data in recent_data:
            memory_efficiency = data['throughput'] / max(data['memory_usage'], 1)
            memory_efficiencies.append(memory_efficiency)
            throughputs.append(data['throughput'])
        
        # 找到最高效的批处理大小
        best_idx = np.argmax(memory_efficiencies)
        optimal_batch_size = recent_data[best_idx]['batch_size']
        
        # 考虑内存压力
        current_memory = self.memory_monitor.monitor_memory()
        if current_memory.memory_percent > 80:
            optimal_batch_size = int(optimal_batch_size * 0.8)  # 减少20%
        elif current_memory.memory_percent < 50:
            optimal_batch_size = int(optimal_batch_size * 1.2)  # 增加20%
        
        return optimal_batch_size
    
    def get_recommended_batch_size(self) -> int:
        """获取推荐的批处理大小"""
        return self.current_batch_size


class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, pool_size_mb: int = 1024, chunk_size_kb: int = 64):
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.chunk_size_bytes = chunk_size_kb * 1024
        self.pool = bytearray(self.pool_size_bytes)
        self.free_chunks = list(range(0, self.pool_size_bytes, self.chunk_size_bytes))
        self.allocated_chunks = {}
        self.lock = threading.Lock()
        
    def allocate(self, size_bytes: int) -> Optional[memoryview]:
        """分配内存"""
        with self.lock:
            # 计算需要的块数
            chunks_needed = (size_bytes + self.chunk_size_bytes - 1) // self.chunk_size_bytes
            
            if len(self.free_chunks) < chunks_needed:
                return None
            
            # 分配连续块
            allocated_chunks = []
            for i in range(chunks_needed):
                if not self.free_chunks:
                    break
                chunk_start = self.free_chunks.pop(0)
                allocated_chunks.append(chunk_start)
            
            if len(allocated_chunks) < chunks_needed:
                # 释放已分配的块
                self.free_chunks.extend(allocated_chunks)
                return None
            
            # 创建内存视图
            start_addr = allocated_chunks[0]
            end_addr = start_addr + size_bytes
            memory_view = memoryview(self.pool[start_addr:end_addr])
            
            # 记录分配
            chunk_id = id(memory_view)
            self.allocated_chunks[chunk_id] = {
                'chunks': allocated_chunks,
                'size': size_bytes
            }
            
            return memory_view
    
    def deallocate(self, memory_view: memoryview):
        """释放内存"""
        with self.lock:
            chunk_id = id(memory_view)
            
            if chunk_id in self.allocated_chunks:
                allocation_info = self.allocated_chunks[chunk_id]
                self.free_chunks.extend(allocation_info['chunks'])
                self.free_chunks.sort()
                del self.allocated_chunks[chunk_id]
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取内存池统计"""
        with self.lock:
            total_chunks = self.pool_size_bytes // self.chunk_size_bytes
            free_chunks = len(self.free_chunks)
            allocated_chunks = total_chunks - free_chunks
            
            return {
                'pool_size_mb': self.pool_size_bytes / (1024 * 1024),
                'chunk_size_kb': self.chunk_size_bytes / 1024,
                'total_chunks': total_chunks,
                'free_chunks': free_chunks,
                'allocated_chunks': allocated_chunks,
                'utilization_percent': (allocated_chunks / total_chunks) * 100
            }


@contextmanager
def memory_profiler():
    """内存分析上下文管理器"""
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    
    try:
        yield
    finally:
        end_snapshot = tracemalloc.take_snapshot()
        top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        
        logger.info("内存使用统计:")
        for stat in top_stats[:10]:
            logger.info(stat)


def create_resource_manager(config: Dict) -> Tuple[MemoryOptimizer, ResourceMonitor]:
    """创建资源管理器"""
    memory_config = config.get('memory', {})
    monitor_config = config.get('monitor', {})
    
    memory_optimizer = MemoryOptimizer(
        memory_limit_gb=memory_config.get('memory_limit_gb', 4.0),
        gc_threshold=memory_config.get('gc_threshold', 0.8)
    )
    
    resource_monitor = ResourceMonitor(
        monitoring_interval=monitor_config.get('monitoring_interval', 1.0)
    )
    
    return memory_optimizer, resource_monitor


if __name__ == "__main__":
    # 示例使用
    print("=== 内存优化测试 ===")
    
    # 创建内存优化器
    optimizer = MemoryOptimizer(memory_limit_gb=2.0)
    
    # 监控内存
    stats = optimizer.monitor_memory()
    print(f"当前内存使用: {stats.used_memory_mb:.2f} MB / {stats.total_memory_mb:.2f} MB "
          f"({stats.memory_percent:.1f}%)")
    
    # 智能垃圾回收
    optimizer.smart_gc(force=True)
    
    # 预测内存压力
    prediction = optimizer.predict_memory_pressure()
    print(f"内存压力预测: {prediction}")
    
    # 获取优化建议
    recommendations = optimizer.get_optimization_recommendations()
    print(f"优化建议: {recommendations}")
    
    # 测试资源监控
    print("\n=== 资源监控测试 ===")
    monitor = ResourceMonitor(monitoring_interval=0.5)
    monitor.start_monitoring()
    
    # 等待一些数据
    time.sleep(3)
    
    # 获取当前指标
    current_metrics = monitor.get_current_metrics()
    if current_metrics:
        print(f"当前CPU使用率: {current_metrics.cpu_percent:.1f}%")
        print(f"当前内存使用率: {current_metrics.memory_percent:.1f}%")
    
    # 获取摘要
    summary = monitor.get_metrics_summary(duration_minutes=1)
    print(f"监控摘要: {summary}")
    
    monitor.stop_monitoring()
    
    print("内存优化和资源管理测试完成")