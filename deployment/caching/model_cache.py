"""
模型缓存和预加载策略模块
实现智能缓存管理和模型预加载优化
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import OrderedDict, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
from sklearn.metrics import mean_squared_error
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCache:
    """智能模型缓存管理器"""
    
    def __init__(self, cache_dir: str = "./model_cache", max_cache_size_gb: float = 2.0):
        self.cache_dir = cache_dir
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.cache_index = {}
        self.access_history = OrderedDict()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size': 0
        }
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载缓存索引
        self._load_cache_index()
        
        # 启动缓存清理线程
        self._start_cleanup_thread()
    
    def _load_cache_index(self):
        """加载缓存索引"""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    self.cache_index = data.get('index', {})
                    self.cache_stats = data.get('stats', self.cache_stats)
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {e}")
    
    def _save_cache_index(self):
        """保存缓存索引"""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        try:
            data = {
                'index': self.cache_index,
                'stats': self.cache_stats
            }
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")
    
    def _calculate_model_hash(self, model: nn.Module, config: Dict = None) -> str:
        """计算模型哈希值"""
        model_hash = hashlib.md5()
        
        # 序列化模型参数
        for name, param in model.state_dict().items():
            model_hash.update(str(param.shape).encode())
            model_hash.update(param.cpu().numpy().tobytes())
        
        # 序列化配置
        if config:
            config_str = json.dumps(config, sort_keys=True)
            model_hash.update(config_str.encode())
        
        return model_hash.hexdigest()
    
    def cache_model(self, model: nn.Module, model_id: str, config: Dict = None) -> str:
        """缓存模型"""
        model_hash = self._calculate_model_hash(model, config)
        cache_key = f"{model_id}_{model_hash}"
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pth")
        
        # 检查是否已缓存
        if cache_key in self.cache_index:
            logger.info(f"模型 {model_id} 已缓存")
            self._update_access_history(cache_key)
            return cache_key
        
        try:
            # 保存模型
            model_data = {
                'model_state_dict': model.state_dict(),
                'config': config or {},
                'timestamp': time.time(),
                'model_id': model_id
            }
            
            torch.save(model_data, cache_file)
            
            # 更新索引
            file_size = os.path.getsize(cache_file)
            self.cache_index[cache_key] = {
                'file_path': cache_file,
                'file_size': file_size,
                'timestamp': time.time(),
                'model_id': model_id,
                'config': config or {}
            }
            
            self.cache_stats['total_size'] += file_size
            
            # 检查缓存大小限制
            self._enforce_cache_limit()
            
            self._save_cache_index()
            logger.info(f"模型 {model_id} 已缓存，缓存键: {cache_key}")
            
            return cache_key
            
        except Exception as e:
            logger.error(f"缓存模型失败: {e}")
            return None
    
    def load_model(self, model: nn.Module, model_id: str, config: Dict = None) -> bool:
        """从缓存加载模型"""
        model_hash = self._calculate_model_hash(model, config)
        cache_key = f"{model_id}_{model_hash}"
        
        if cache_key in self.cache_index:
            try:
                cache_info = self.cache_index[cache_key]
                cache_file = cache_info['file_path']
                
                if os.path.exists(cache_file):
                    # 加载模型数据
                    model_data = torch.load(cache_file, map_location='cpu')
                    model.load_state_dict(model_data['model_state_dict'])
                    
                    self._update_access_history(cache_key)
                    self.cache_stats['hits'] += 1
                    
                    logger.info(f"从缓存加载模型 {model_id} 成功")
                    return True
                else:
                    # 缓存文件不存在，清理索引
                    del self.cache_index[cache_key]
                    self.cache_stats['misses'] += 1
                    
            except Exception as e:
                logger.error(f"从缓存加载模型失败: {e}")
                self.cache_stats['misses'] += 1
        else:
            self.cache_stats['misses'] += 1
        
        return False
    
    def _update_access_history(self, cache_key: str):
        """更新访问历史"""
        if cache_key in self.access_history:
            del self.access_history[cache_key]
        
        self.access_history[cache_key] = time.time()
        
        # 保持访问历史大小限制
        if len(self.access_history) > 1000:
            oldest_key = next(iter(self.access_history))
            del self.access_history[oldest_key]
    
    def _enforce_cache_limit(self):
        """执行缓存大小限制"""
        if self.cache_stats['total_size'] <= self.max_cache_size_bytes:
            return
        
        # 使用LRU策略清理缓存
        sorted_items = sorted(self.access_history.items(), key=lambda x: x[1])
        
        while self.cache_stats['total_size'] > self.max_cache_size_bytes and sorted_items:
            oldest_key, _ = sorted_items.pop(0)
            
            if oldest_key in self.cache_index:
                cache_info = self.cache_index[oldest_key]
                file_path = cache_info['file_path']
                
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    self.cache_stats['total_size'] -= cache_info['file_size']
                    del self.cache_index[oldest_key]
                    self.cache_stats['evictions'] += 1
                    
                except Exception as e:
                    logger.warning(f"删除缓存文件失败: {e}")
    
    def _start_cleanup_thread(self):
        """启动缓存清理线程"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # 每5分钟清理一次
                    self._enforce_cache_limit()
                    self._save_cache_index()
                except Exception as e:
                    logger.error(f"缓存清理失败: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        hit_rate = 0
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_requests > 0:
            hit_rate = self.cache_stats['hits'] / total_requests
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'cache_size_gb': self.cache_stats['total_size'] / (1024 * 1024 * 1024),
            'max_cache_size_gb': self.max_cache_size_bytes / (1024 * 1024 * 1024),
            'cached_models': len(self.cache_index)
        }
    
    def clear_cache(self):
        """清空缓存"""
        for cache_key, cache_info in self.cache_index.items():
            try:
                file_path = cache_info['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"删除缓存文件失败: {e}")
        
        self.cache_index.clear()
        self.access_history.clear()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size': 0
        }
        
        self._save_cache_index()
        logger.info("缓存已清空")


class ModelPreloader:
    """模型预加载器"""
    
    def __init__(self, max_concurrent_loads: int = 3):
        self.max_concurrent_loads = max_concurrent_loads
        self.preloaded_models = {}
        self.loading_queue = []
        self.loading_thread = None
        self.stop_loading = False
        
        # 启动预加载线程
        self._start_preloader()
    
    def _start_preloader(self):
        """启动预加载线程"""
        def preloader_worker():
            while not self.stop_loading:
                try:
                    if self.loading_queue:
                        model_info = self.loading_queue.pop(0)
                        self._load_single_model(model_info)
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"预加载失败: {e}")
        
        self.loading_thread = threading.Thread(target=preloader_worker, daemon=True)
        self.loading_thread.start()
    
    def _load_single_model(self, model_info: Dict):
        """加载单个模型"""
        try:
            model_id = model_info['model_id']
            model_class = model_info['model_class']
            model_config = model_info.get('config', {})
            
            # 创建模型实例
            model = model_class(**model_config)
            
            # 加载权重
            if 'weights_path' in model_info:
                model.load_state_dict(torch.load(model_info['weights_path'], map_location='cpu'))
            
            # 缓存模型
            self.preloaded_models[model_id] = {
                'model': model,
                'config': model_config,
                'load_time': time.time()
            }
            
            logger.info(f"模型 {model_id} 预加载完成")
            
        except Exception as e:
            logger.error(f"预加载模型失败: {e}")
    
    def preload_model(self, model_id: str, model_class: type, model_config: Dict = None, 
                     weights_path: str = None):
        """预加载模型"""
        model_info = {
            'model_id': model_id,
            'model_class': model_class,
            'config': model_config or {},
        }
        
        if weights_path:
            model_info['weights_path'] = weights_path
        
        self.loading_queue.append(model_info)
        logger.info(f"模型 {model_id} 已加入预加载队列")
    
    def get_preloaded_model(self, model_id: str) -> Optional[nn.Module]:
        """获取预加载的模型"""
        if model_id in self.preloaded_models:
            model_info = self.preloaded_models[model_id]
            logger.info(f"获取预加载模型 {model_id}")
            return model_info['model']
        
        return None
    
    def preload_batch(self, model_configs: List[Dict]):
        """批量预加载模型"""
        with ThreadPoolExecutor(max_workers=self.max_concurrent_loads) as executor:
            futures = []
            
            for config in model_configs:
                future = executor.submit(
                    self._load_single_model, config
                )
                futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"批量预加载失败: {e}")
    
    def get_preload_stats(self) -> Dict:
        """获取预加载统计信息"""
        return {
            'preloaded_models': len(self.preloaded_models),
            'loading_queue_size': len(self.loading_queue),
            'model_ids': list(self.preloaded_models.keys())
        }
    
    def stop(self):
        """停止预加载器"""
        self.stop_loading = True
        if self.loading_thread:
            self.loading_thread.join(timeout=1.0)


class AdaptiveCacheStrategy:
    """自适应缓存策略"""
    
    def __init__(self, base_cache: ModelCache):
        self.base_cache = base_cache
        self.access_patterns = defaultdict(list)
        self.prediction_window = 100  # 预测窗口大小
        self.confidence_threshold = 0.7
        
    def record_access(self, model_id: str, access_time: float):
        """记录访问模式"""
        self.access_patterns[model_id].append(access_time)
        
        # 保持历史记录大小
        if len(self.access_patterns[model_id]) > 1000:
            self.access_patterns[model_id] = self.access_patterns[model_id][-500:]
    
    def predict_next_access(self, model_id: str, current_time: float) -> Tuple[float, float]:
        """预测下次访问时间和置信度"""
        if model_id not in self.access_patterns or len(self.access_patterns[model_id]) < 2:
            return current_time + 3600, 0.0  # 默认1小时后，置信度0
        
        access_times = self.access_patterns[model_id]
        
        # 计算访问间隔
        intervals = []
        for i in range(1, len(access_times)):
            interval = access_times[i] - access_times[i-1]
            intervals.append(interval)
        
        if not intervals:
            return current_time + 3600, 0.0
        
        # 计算平均间隔和标准差
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # 预测下次访问
        predicted_time = current_time + avg_interval
        
        # 计算置信度（基于标准差）
        if std_interval > 0:
            confidence = max(0.0, min(1.0, 1.0 - (std_interval / avg_interval)))
        else:
            confidence = 1.0
        
        return predicted_time, confidence
    
    def get_preload_recommendations(self, current_time: float) -> List[str]:
        """获取预加载推荐"""
        recommendations = []
        
        for model_id in self.base_cache.cache_index.keys():
            predicted_time, confidence = self.predict_next_access(model_id, current_time)
            
            if confidence > self.confidence_threshold:
                time_until_access = predicted_time - current_time
                if 0 < time_until_access < 1800:  # 30分钟内会访问
                    recommendations.append(model_id)
        
        return recommendations
    
    def optimize_cache_priority(self):
        """优化缓存优先级"""
        current_time = time.time()
        recommendations = self.get_preload_recommendations(current_time)
        
        # 这里可以实现更复杂的缓存优化逻辑
        logger.info(f"缓存优化推荐: {recommendations}")
        
        return recommendations


class MemoryOptimizedLoader:
    """内存优化的模型加载器"""
    
    def __init__(self, memory_limit_gb: float = 4.0):
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.loaded_models = {}
        self.memory_usage = 0
        self.load_order = []
        
    def load_model_with_memory_management(self, model_id: str, model: nn.Module, 
                                        weights_path: str = None) -> bool:
        """内存管理的模型加载"""
        try:
            # 检查内存使用
            current_memory = psutil.virtual_memory().used
            available_memory = psutil.virtual_memory().available
            
            if available_memory < self.memory_limit_bytes * 0.2:  # 可用内存少于20%
                self._evict_least_recently_used()
            
            # 加载模型权重
            if weights_path:
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
            
            # 估算模型大小
            model_size = self._estimate_model_size(model)
            
            if self.memory_usage + model_size > self.memory_limit_bytes:
                logger.warning(f"模型 {model_id} 过大，跳过加载")
                return False
            
            # 加载模型
            self.loaded_models[model_id] = {
                'model': model,
                'size': model_size,
                'load_time': time.time()
            }
            
            self.memory_usage += model_size
            self.load_order.append(model_id)
            
            logger.info(f"模型 {model_id} 加载完成，内存使用: {model_size / 1024 / 1024:.2f} MB")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def _estimate_model_size(self, model: nn.Module) -> int:
        """估算模型大小（字节）"""
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        return total_size
    
    def _evict_least_recently_used(self):
        """驱逐最近最少使用的模型"""
        if not self.load_order:
            return
        
        # 找到最久未使用的模型
        lru_model_id = self.load_order.pop(0)
        
        if lru_model_id in self.loaded_models:
            model_info = self.loaded_models[lru_model_id]
            self.memory_usage -= model_info['size']
            del self.loaded_models[lru_model_id]
            
            logger.info(f"驱逐模型 {lru_model_id}")
            
            # 强制垃圾回收
            gc.collect()
    
    def get_memory_stats(self) -> Dict:
        """获取内存使用统计"""
        return {
            'loaded_models': len(self.loaded_models),
            'memory_usage_mb': self.memory_usage / 1024 / 1024,
            'memory_limit_gb': self.memory_limit_bytes / 1024 / 1024 / 1024,
            'memory_usage_percent': (self.memory_usage / self.memory_limit_bytes) * 100,
            'model_ids': list(self.loaded_models.keys())
        }


def create_cache_manager(config: Dict) -> ModelCache:
    """创建缓存管理器"""
    cache_config = config.get('cache', {})
    cache_dir = cache_config.get('cache_dir', './model_cache')
    max_size_gb = cache_config.get('max_cache_size_gb', 2.0)
    
    return ModelCache(cache_dir, max_size_gb)


def create_preloader(config: Dict) -> ModelPreloader:
    """创建预加载器"""
    preload_config = config.get('preload', {})
    max_concurrent = preload_config.get('max_concurrent_loads', 3)
    
    return ModelPreloader(max_concurrent)


if __name__ == "__main__":
    # 示例使用
    class SimpleModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=50):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    # 创建缓存管理器
    cache_manager = ModelCache("./test_cache", max_cache_size_gb=1.0)
    
    # 创建模型
    model = SimpleModel()
    
    # 测试缓存
    print("=== 缓存测试 ===")
    cache_key = cache_manager.cache_model(model, "simple_model", {"input_size": 10})
    print(f"缓存键: {cache_key}")
    
    # 测试加载
    new_model = SimpleModel()
    success = cache_manager.load_model(new_model, "simple_model", {"input_size": 10})
    print(f"加载成功: {success}")
    
    # 获取统计信息
    stats = cache_manager.get_cache_stats()
    print(f"缓存统计: {stats}")
    
    # 测试预加载器
    print("\n=== 预加载测试 ===")
    preloader = ModelPreloader(max_concurrent_loads=2)
    preloader.preload_model("test_model", SimpleModel, {"input_size": 10})
    
    # 等待预加载完成
    time.sleep(2)
    
    preload_stats = preloader.get_preload_stats()
    print(f"预加载统计: {preload_stats}")
    
    # 清理
    cache_manager.clear_cache()
    preloader.stop()
    
    print("缓存和预加载测试完成")