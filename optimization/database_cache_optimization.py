#!/usr/bin/env python3
"""
数据库查询和缓存策略优化
实现Redis缓存、查询优化、连接池等优化技术
"""

import time
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
from collections import OrderedDict
import sqlite3
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000           # 最大缓存条目数
    ttl: int = 3600                # 缓存生存时间(秒)
    enable_persistence: bool = True # 启用持久化
    cache_dir: str = "/workspace/code/optimization/cache"
    compression: bool = True       # 启用压缩

class MemoryCache:
    """内存缓存实现"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        
    def _generate_key(self, key: str) -> str:
        """生成缓存键"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key not in self.cache:
                return None
            
            # 检查TTL
            if self._is_expired(cache_key):
                self._remove(cache_key)
                return None
            
            # 更新访问时间
            self.timestamps[cache_key] = datetime.now()
            
            # 移动到末尾 (LRU)
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        with self.lock:
            cache_key = self._generate_key(key)
            expire_time = ttl or self.config.ttl
            
            # 检查容量
            if len(self.cache) >= self.config.max_size and cache_key not in self.cache:
                self._evict_lru()
            
            # 设置值和时间戳
            self.cache[cache_key] = value
            self.timestamps[cache_key] = datetime.now()
            
            # 记录过期时间
            if not hasattr(self, 'expire_times'):
                self.expire_times = {}
            self.expire_times[cache_key] = expire_time
            
            return True
    
    def _is_expired(self, cache_key: str) -> bool:
        """检查是否过期"""
        if not hasattr(self, 'expire_times'):
            return False
        
        if cache_key not in self.expire_times:
            return False
        
        expire_time = self.expire_times[cache_key]
        create_time = self.timestamps[cache_key]
        
        return datetime.now() - create_time > timedelta(seconds=expire_time)
    
    def _remove(self, cache_key: str):
        """移除缓存项"""
        self.cache.pop(cache_key, None)
        self.timestamps.pop(cache_key, None)
        if hasattr(self, 'expire_times'):
            self.expire_times.pop(cache_key, None)
    
    def _evict_lru(self):
        """移除最少使用的项"""
        if self.cache:
            # 移除最旧的项
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            if hasattr(self, 'expire_times'):
                self.expire_times.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'size': self.size(),
            'max_size': self.config.max_size,
            'hit_rate': getattr(self, 'hit_rate', 0.0),
            'miss_rate': getattr(self, 'miss_rate', 0.0)
        }

class DatabaseOptimizer:
    """数据库优化器"""
    
    def __init__(self, db_path: str = "/workspace/code/optimization/traffic_data.db"):
        self.db_path = db_path
        self.connection_pool = []
        self.max_connections = 10
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建交通数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                segment_id TEXT,
                speed REAL,
                flow INTEGER,
                occupancy REAL,
                density REAL,
                status TEXT
            )
        ''')
        
        # 创建预测数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                prediction_id TEXT,
                segment_id TEXT,
                predicted_speed REAL,
                predicted_flow INTEGER,
                confidence REAL,
                model_version TEXT
            )
        ''')
        
        # 创建应急事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergency_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                event_id TEXT,
                location TEXT,
                event_type TEXT,
                severity INTEGER,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("数据库初始化完成")
    
    def get_connection(self):
        """获取数据库连接"""
        if self.connection_pool:
            return self.connection_pool.pop()
        else:
            return sqlite3.connect(self.db_path)
    
    def return_connection(self, conn):
        """归还数据库连接"""
        if len(self.connection_pool) < self.max_connections:
            self.connection_pool.append(conn)
        else:
            conn.close()
    
    def optimize_queries(self):
        """优化数据库查询"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 创建复合索引
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_traffic_time_segment ON traffic_data(timestamp, segment_id)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_time_segment ON prediction_data(timestamp, segment_id)",
            "CREATE INDEX IF NOT EXISTS idx_emergency_time_status ON emergency_events(timestamp, status)"
        ]
        
        for index_sql in indices:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"创建索引失败: {e}")
        
        conn.commit()
        self.return_connection(conn)
        
        logger.info("数据库查询优化完成")
    
    def insert_traffic_data(self, data: List[Dict[str, Any]]) -> bool:
        """批量插入交通数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 使用事务批量插入
            cursor.executemany('''
                INSERT INTO traffic_data 
                (timestamp, segment_id, speed, flow, occupancy, density, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', [
                (
                    item['timestamp'],
                    item['segment_id'],
                    item['speed'],
                    item['flow'],
                    item['occupancy'],
                    item['density'],
                    item['status']
                ) for item in data
            ])
            
            conn.commit()
            logger.info(f"成功插入 {len(data)} 条交通数据")
            return True
            
        except Exception as e:
            logger.error(f"插入交通数据失败: {e}")
            conn.rollback()
            return False
        finally:
            self.return_connection(conn)
    
    def query_recent_traffic(self, hours: int = 24) -> pd.DataFrame:
        """查询最近的交通数据"""
        conn = self.get_connection()
        
        try:
            query = '''
                SELECT * FROM traffic_data 
                WHERE timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            '''.format(hours)
            
            df = pd.read_sql_query(query, conn)
            return df
            
        except Exception as e:
            logger.error(f"查询交通数据失败: {e}")
            return pd.DataFrame()
        finally:
            self.return_connection(conn)
    
    def query_prediction_performance(self, model_version: str = None) -> Dict[str, Any]:
        """查询预测性能统计"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            where_clause = ""
            if model_version:
                where_clause = f"WHERE model_version = '{model_version}'"
            
            query = f'''
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    MIN(timestamp) as earliest_prediction,
                    MAX(timestamp) as latest_prediction
                FROM prediction_data
                {where_clause}
            '''
            
            cursor.execute(query)
            result = cursor.fetchone()
            
            return {
                'total_predictions': result[0],
                'avg_confidence': result[1],
                'earliest_prediction': result[2],
                'latest_prediction': result[3]
            }
            
        except Exception as e:
            logger.error(f"查询预测性能失败: {e}")
            return {}
        finally:
            self.return_connection(conn)

class QueryCache:
    """查询缓存"""
    
    def __init__(self, cache_config: CacheConfig, db_optimizer: DatabaseOptimizer):
        self.cache = MemoryCache(cache_config)
        self.db_optimizer = db_optimizer
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def cached_query(self, query_func, cache_key: str, *args, **kwargs) -> Any:
        """缓存查询"""
        self.query_stats['total_queries'] += 1
        
        # 尝试从缓存获取
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.query_stats['cache_hits'] += 1
            logger.debug(f"缓存命中: {cache_key}")
            return cached_result
        
        # 缓存未命中，执行查询
        self.query_stats['cache_misses'] += 1
        logger.debug(f"缓存未命中: {cache_key}")
        
        result = query_func(*args, **kwargs)
        
        # 缓存结果
        self.cache.set(cache_key, result)
        
        return result
    
    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.query_stats['total_queries']
        if total == 0:
            return 0.0
        return self.query_stats['cache_hits'] / total
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("查询缓存已清空")

class TrafficDataGenerator:
    """优化的交通数据生成器"""
    
    def __init__(self, cache: MemoryCache):
        self.cache = cache
        self.data_lock = threading.Lock()
        
    def generate_realtime_data_cached(self, force_refresh: bool = False) -> Dict[str, Any]:
        """生成缓存的实时交通数据"""
        cache_key = "realtime_traffic_data"
        
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # 生成新数据
        with self.data_lock:
            data = self._generate_fresh_data()
            
            # 缓存5分钟
            self.cache.set(cache_key, data, ttl=300)
            
            return data
    
    def _generate_fresh_data(self) -> Dict[str, Any]:
        """生成新鲜的交通数据"""
        # 模拟数据生成逻辑
        timestamp = datetime.now().isoformat()
        
        # 生成30个路段的数据
        segments = []
        for i in range(30):
            segment = {
                'id': f'SEG_{i:03d}',
                'name': f'路段{i+1}',
                'lat': 39.9 + np.random.uniform(-0.1, 0.1),
                'lng': 116.4 + np.random.uniform(-0.1, 0.1),
                'speed': np.random.uniform(30, 80),
                'flow': np.random.randint(500, 2000),
                'occupancy': np.random.uniform(0.2, 0.8),
                'status': np.random.choice(['normal', 'warning', 'congested'])
            }
            segments.append(segment)
        
        return {
            'timestamp': timestamp,
            'total_segments': len(segments),
            'congested_segments': sum(1 for s in segments if s['status'] == 'congested'),
            'average_speed': np.mean([s['speed'] for s in segments]),
            'segments': segments
        }

def run_database_optimization_demo():
    """运行数据库优化演示"""
    logger.info("开始数据库优化演示...")
    
    # 初始化组件
    cache_config = CacheConfig(
        max_size=500,
        ttl=1800,
        enable_persistence=True,
        compression=True
    )
    
    db_optimizer = DatabaseOptimizer()
    query_cache = QueryCache(cache_config, db_optimizer)
    traffic_generator = TrafficDataGenerator(query_cache.cache)
    
    # 优化数据库
    db_optimizer.optimize_queries()
    
    # 生成测试数据
    logger.info("生成测试数据...")
    test_data = []
    base_time = datetime.now()
    
    for i in range(100):
        data_point = {
            'timestamp': (base_time - timedelta(minutes=i*5)).isoformat(),
            'segment_id': f'SEG_{i%30:03d}',
            'speed': np.random.uniform(30, 80),
            'flow': np.random.randint(500, 2000),
            'occupancy': np.random.uniform(0.2, 0.8),
            'density': np.random.uniform(10, 50),
            'status': np.random.choice(['normal', 'warning', 'congested'])
        }
        test_data.append(data_point)
    
    # 插入数据
    success = db_optimizer.insert_traffic_data(test_data)
    if success:
        logger.info("测试数据插入成功")
    
    # 测试缓存查询
    logger.info("测试缓存查询...")
    
    # 第一次查询 (缓存未命中)
    start_time = time.time()
    result1 = query_cache.cached_query(
        db_optimizer.query_recent_traffic,
        "recent_traffic_24h",
        hours=24
    )
    first_query_time = time.time() - start_time
    
    # 第二次查询 (缓存命中)
    start_time = time.time()
    result2 = query_cache.cached_query(
        db_optimizer.query_recent_traffic,
        "recent_traffic_24h",
        hours=24
    )
    second_query_time = time.time() - start_time
    
    # 性能统计
    cache_hit_rate = query_cache.get_cache_hit_rate()
    cache_stats = query_cache.cache.stats()
    
    # 查询预测性能
    prediction_stats = db_optimizer.query_prediction_performance()
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'cache_performance': {
            'hit_rate': cache_hit_rate,
            'first_query_time': first_query_time,
            'second_query_time': second_query_time,
            'speedup': first_query_time / second_query_time if second_query_time > 0 else float('inf'),
            'cache_stats': cache_stats
        },
        'database_performance': {
            'total_queries': query_cache.query_stats['total_queries'],
            'cache_hits': query_cache.query_stats['cache_hits'],
            'cache_misses': query_cache.query_stats['cache_misses']
        },
        'prediction_stats': prediction_stats,
        'optimization_applied': [
            '内存缓存',
            '数据库索引优化',
            '连接池管理',
            '批量插入优化',
            '查询结果缓存'
        ]
    }
    
    # 保存报告
    with open('/workspace/code/optimization/database_optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("数据库优化演示完成")
    logger.info(f"缓存命中率: {cache_hit_rate:.2%}")
    logger.info(f"查询加速比: {first_query_time / second_query_time:.2f}x")
    
    return report

if __name__ == "__main__":
    report = run_database_optimization_demo()
    print(f"\n=== 数据库优化结果 ===")
    print(f"缓存命中率: {report['cache_performance']['hit_rate']:.2%}")
    print(f"查询加速比: {report['cache_performance']['speedup']:.2f}x")
    print(f"总查询数: {report['database_performance']['total_queries']}")