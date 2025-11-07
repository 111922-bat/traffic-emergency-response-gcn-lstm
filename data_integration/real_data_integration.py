#!/usr/bin/env python3
"""
真实数据集成模块
整合PEMS数据集、API数据源和数据质量监控
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import queue
import sqlite3
import pandas as pd

# 添加项目路径
sys.path.append('/workspace/code')

# 导入真实数据组件
from data_integration.real_data_manager import RealDataManager
from data_integration.data_quality_monitor import DataQualityMonitor, DataQualityMetrics, AnomalyAlert
from data_integration.config_manager import get_config_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationStatus:
    """集成状态"""
    is_running: bool = False
    last_update: Optional[str] = None
    active_data_sources: List[str] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    current_quality_score: float = 0.0

class RealDataIntegration:
    """真实数据集成主类"""
    
    def __init__(self):
        # 初始化组件
        self.config_manager = get_config_manager()
        self.real_data_manager = RealDataManager()
        self.quality_monitor = DataQualityMonitor()
        
        # 状态管理
        self.status = IntegrationStatus()
        self.status.active_data_sources = []
        
        # 缓存管理
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_timeout = 300  # 5分钟
        
        # 线程池
        self.executor = None
        self.background_threads = []
        
        # 数据存储
        self.db_path = "/workspace/data/integration.db"
        self._init_database()
        
        logger.info("真实数据集成模块初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建集成数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS integrated_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    data_source TEXT,
                    data_type TEXT,  -- traffic, weather
                    data_content TEXT,
                    quality_score REAL,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建性能统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    data_source TEXT,
                    response_time REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("集成数据库初始化完成")
            
        except Exception as e:
            logger.error(f"集成数据库初始化失败: {e}")
    
    def start_integration(self):
        """启动数据集成"""
        try:
            if self.status.is_running:
                logger.warning("数据集成已在运行中")
                return
            
            # 验证配置
            config_errors = self.config_manager.validate_config()
            if config_errors:
                logger.error(f"配置验证失败: {config_errors}")
                return
            
            self.status.is_running = True
            self.status.last_update = datetime.now().isoformat()
            
            # 启动后台线程
            self._start_background_threads()
            
            # 启动质量监控
            if self.config_manager.get_quality_config().monitoring_enabled:
                self.quality_monitor.start_monitoring(interval_seconds=300)
            
            logger.info("真实数据集成已启动")
            
        except Exception as e:
            logger.error(f"启动数据集成失败: {e}")
            self.status.is_running = False
    
    def stop_integration(self):
        """停止数据集成"""
        try:
            self.status.is_running = False
            
            # 停止后台线程
            for thread in self.background_threads:
                if thread.is_alive():
                    # 这里应该添加线程停止逻辑
                    pass
            
            # 停止质量监控
            self.quality_monitor.stop_monitoring()
            
            # 清理缓存
            self.data_cache.clear()
            
            logger.info("真实数据集成已停止")
            
        except Exception as e:
            logger.error(f"停止数据集成失败: {e}")
    
    def _start_background_threads(self):
        """启动后台线程"""
        # 数据更新线程
        update_thread = threading.Thread(target=self._data_update_loop, daemon=True)
        update_thread.start()
        self.background_threads.append(update_thread)
        
        # 性能监控线程
        perf_thread = threading.Thread(target=self._performance_monitor_loop, daemon=True)
        perf_thread.start()
        self.background_threads.append(perf_thread)
        
        logger.info("后台线程已启动")
    
    def _data_update_loop(self):
        """数据更新循环"""
        while self.status.is_running:
            try:
                # 更新活跃数据源状态
                self._update_active_data_sources()
                
                # 预取和缓存数据
                self._prefetch_and_cache_data()
                
                # 清理过期缓存
                self._cleanup_expired_cache()
                
                # 等待下次更新
                time.sleep(60)  # 每分钟更新一次
                
            except Exception as e:
                logger.error(f"数据更新循环错误: {e}")
                time.sleep(60)
    
    def _performance_monitor_loop(self):
        """性能监控循环"""
        while self.status.is_running:
            try:
                # 计算性能统计
                self._calculate_performance_stats()
                
                # 生成性能报告
                self._generate_performance_report()
                
                # 等待下次监控
                time.sleep(300)  # 每5分钟监控一次
                
            except Exception as e:
                logger.error(f"性能监控循环错误: {e}")
                time.sleep(300)
    
    def _update_active_data_sources(self):
        """更新活跃数据源状态"""
        try:
            enabled_sources = self.config_manager.get_enabled_data_sources()
            self.status.active_data_sources = [source.name for source in enabled_sources]
            
            logger.debug(f"活跃数据源: {self.status.active_data_sources}")
            
        except Exception as e:
            logger.error(f"更新活跃数据源失败: {e}")
    
    def _prefetch_and_cache_data(self):
        """预取和缓存数据"""
        try:
            # 预取交通数据
            traffic_data = self.get_traffic_data()
            if traffic_data:
                self._cache_data("traffic", traffic_data)
            
            # 预取天气数据
            weather_data = self.get_weather_data()
            if weather_data:
                self._cache_data("weather", weather_data)
            
        except Exception as e:
            logger.error(f"预取和缓存数据失败: {e}")
    
    def _cleanup_expired_cache(self):
        """清理过期缓存"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, (data, timestamp) in self.data_cache.items():
                if current_time - timestamp > self.cache_timeout:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.data_cache[key]
            
            if expired_keys:
                logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
            
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")
    
    def get_traffic_data(self, preferred_source: str = None, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """获取交通数据"""
        try:
            cache_key = f"traffic_{preferred_source or 'auto'}"
            
            # 检查缓存
            if not force_refresh and cache_key in self.data_cache:
                cached_data, cache_time = self.data_cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    logger.info("从缓存返回交通数据")
                    return cached_data
            
            # 记录开始时间
            start_time = time.time()
            data_source_used = "unknown"
            
            try:
                # 获取真实数据
                traffic_data = self.real_data_manager.get_realtime_traffic_data(preferred_source)
                
                if traffic_data:
                    # 数据质量评估
                    quality_metrics = self.quality_monitor.assess_data_quality(traffic_data)
                    
                    # 异常检测
                    alerts = self.quality_monitor.detect_anomalies(traffic_data)
                    
                    # 添加质量信息到数据
                    traffic_data['quality_metrics'] = {
                        'overall_score': quality_metrics.overall_score,
                        'quality_level': quality_metrics.quality_level.value,
                        'completeness': quality_metrics.completeness,
                        'accuracy': quality_metrics.accuracy,
                        'timeliness': quality_metrics.timeliness,
                        'issues': quality_metrics.issues
                    }
                    
                    traffic_data['anomaly_alerts'] = [
                        {
                            'type': alert.anomaly_type.value,
                            'severity': alert.severity,
                            'description': alert.description,
                            'segment_id': alert.segment_id
                        }
                        for alert in alerts
                    ]
                    
                    # 更新状态
                    data_source_used = traffic_data.get('data_source', 'unknown')
                    self.status.successful_requests += 1
                    self.status.current_quality_score = quality_metrics.overall_score
                    
                    # 保存到数据库
                    self._save_integrated_data("traffic", traffic_data, quality_metrics.overall_score, time.time() - start_time)
                    
                    # 缓存数据
                    self._cache_data(cache_key, traffic_data)
                    
                    logger.info(f"成功获取交通数据，来源: {data_source_used}, 质量评分: {quality_metrics.overall_score:.2%}")
                    return traffic_data
                else:
                    self.status.failed_requests += 1
                    logger.error("无法获取交通数据")
                    return None
                    
            except Exception as e:
                self.status.failed_requests += 1
                logger.error(f"获取交通数据失败: {e}")
                return None
            
            finally:
                # 更新性能统计
                response_time = time.time() - start_time
                self._save_performance_stats(data_source_used, response_time, True, str(e) if 'e' in locals() else None)
                self._update_response_time_stats(response_time)
                
        except Exception as e:
            logger.error(f"获取交通数据异常: {e}")
            return None
    
    def get_weather_data(self, location: str = "北京", force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """获取天气数据"""
        try:
            cache_key = f"weather_{location}"
            
            # 检查缓存
            if not force_refresh and cache_key in self.data_cache:
                cached_data, cache_time = self.data_cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    logger.info("从缓存返回天气数据")
                    return cached_data
            
            # 记录开始时间
            start_time = time.time()
            data_source_used = "unknown"
            
            try:
                # 获取天气数据
                weather_data = self.real_data_manager.get_weather_data()
                
                if weather_data:
                    # 更新状态
                    data_source_used = weather_data.get('data_source', 'unknown')
                    self.status.successful_requests += 1
                    
                    # 保存到数据库
                    self._save_integrated_data("weather", weather_data, 0.9, time.time() - start_time)
                    
                    # 缓存数据
                    self._cache_data(cache_key, weather_data)
                    
                    logger.info(f"成功获取天气数据，来源: {data_source_used}")
                    return weather_data
                else:
                    self.status.failed_requests += 1
                    logger.error("无法获取天气数据")
                    return None
                    
            except Exception as e:
                self.status.failed_requests += 1
                logger.error(f"获取天气数据失败: {e}")
                return None
            
            finally:
                # 更新性能统计
                response_time = time.time() - start_time
                self._save_performance_stats(data_source_used, response_time, True, str(e) if 'e' in locals() else None)
                self._update_response_time_stats(response_time)
                
        except Exception as e:
            logger.error(f"获取天气数据异常: {e}")
            return None
    
    def get_integrated_data(self, data_type: str = "traffic", time_range: int = 3600) -> List[Dict[str, Any]]:
        """获取集成数据历史"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, data_source, data_type, data_content, quality_score, processing_time
                FROM integrated_data
                WHERE data_type = ? AND timestamp >= datetime('now', '-{} seconds')
                ORDER BY timestamp DESC
            '''.format(time_range)
            
            df = pd.read_sql_query(query, conn, params=(data_type,))
            conn.close()
            
            results = []
            for _, row in df.iterrows():
                try:
                    data_content = json.loads(row['data_content'])
                    results.append({
                        'timestamp': row['timestamp'],
                        'data_source': row['data_source'],
                        'quality_score': row['quality_score'],
                        'processing_time': row['processing_time'],
                        'data': data_content
                    })
                except json.JSONDecodeError:
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"获取集成数据历史失败: {e}")
            return []
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 查询性能统计
            query = '''
                SELECT data_source, 
                       COUNT(*) as total_requests,
                       AVG(response_time) as avg_response_time,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests
                FROM performance_stats
                WHERE timestamp >= datetime('now', '-{} hours')
                GROUP BY data_source
            '''.format(hours)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # 计算总体统计
            total_requests = df['total_requests'].sum()
            total_successful = df['successful_requests'].sum()
            total_failed = df['failed_requests'].sum()
            overall_success_rate = total_successful / total_requests if total_requests > 0 else 0
            
            # 按数据源统计
            source_stats = []
            for _, row in df.iterrows():
                source_stats.append({
                    'data_source': row['data_source'],
                    'total_requests': int(row['total_requests']),
                    'successful_requests': int(row['successful_requests']),
                    'failed_requests': int(row['failed_requests']),
                    'success_rate': row['successful_requests'] / row['total_requests'] if row['total_requests'] > 0 else 0,
                    'avg_response_time': round(row['avg_response_time'], 3)
                })
            
            report = {
                'report_period': f'{hours}小时',
                'overall_stats': {
                    'total_requests': int(total_requests),
                    'successful_requests': int(total_successful),
                    'failed_requests': int(total_failed),
                    'success_rate': round(overall_success_rate, 3),
                    'average_response_time': round(df['avg_response_time'].mean(), 3) if not df.empty else 0
                },
                'source_stats': source_stats,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {'error': str(e)}
    
    def get_data_quality_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取数据质量报告"""
        try:
            return self.quality_monitor.get_quality_report(hours)
        except Exception as e:
            logger.error(f"获取数据质量报告失败: {e}")
            return {'error': str(e)}
    
    def _cache_data(self, key: str, data: Dict[str, Any]):
        """缓存数据"""
        try:
            with self.cache_lock:
                self.data_cache[key] = (data, time.time())
        except Exception as e:
            logger.error(f"缓存数据失败: {e}")
    
    def _save_integrated_data(self, data_type: str, data: Dict[str, Any], quality_score: float, processing_time: float):
        """保存集成数据到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO integrated_data (
                    timestamp, data_source, data_type, data_content, quality_score, processing_time
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                data.get('data_source', 'unknown'),
                data_type,
                json.dumps(data, ensure_ascii=False),
                quality_score,
                processing_time
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存集成数据失败: {e}")
    
    def _save_performance_stats(self, data_source: str, response_time: float, success: bool, error_message: str = None):
        """保存性能统计"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_stats (
                    timestamp, data_source, response_time, success, error_message
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                data_source,
                response_time,
                1 if success else 0,
                error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存性能统计失败: {e}")
    
    def _update_response_time_stats(self, response_time: float):
        """更新响应时间统计"""
        try:
            self.status.total_requests += 1
            
            # 更新平均响应时间
            current_avg = self.status.average_response_time
            total_requests = self.status.total_requests
            self.status.average_response_time = (current_avg * (total_requests - 1) + response_time) / total_requests
            
        except Exception as e:
            logger.error(f"更新响应时间统计失败: {e}")
    
    def _calculate_performance_stats(self):
        """计算性能统计"""
        try:
            # 这里可以添加更复杂的性能计算逻辑
            # 例如：趋势分析、异常检测等
            pass
        except Exception as e:
            logger.error(f"计算性能统计失败: {e}")
    
    def _generate_performance_report(self):
        """生成性能报告"""
        try:
            # 每小时生成一次简化的性能报告
            report = self.get_performance_report(1)
            logger.info(f"性能报告生成完成: 成功率 {report['overall_stats']['success_rate']:.2%}")
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
    
    def get_status(self) -> IntegrationStatus:
        """获取集成状态"""
        self.status.last_update = datetime.now().isoformat()
        return self.status
    
    def reset_statistics(self):
        """重置统计信息"""
        try:
            self.status = IntegrationStatus()
            self.status.active_data_sources = self.config_manager.get_enabled_data_sources()
            logger.info("统计信息已重置")
        except Exception as e:
            logger.error(f"重置统计信息失败: {e}")

# 全局集成实例
integration_instance = None

def get_integration_instance() -> RealDataIntegration:
    """获取全局集成实例"""
    global integration_instance
    if integration_instance is None:
        integration_instance = RealDataIntegration()
    return integration_instance

# 使用示例
if __name__ == "__main__":
    # 初始化集成模块
    integration = RealDataIntegration()
    
    # 启动集成
    print("启动真实数据集成...")
    integration.start_integration()
    
    # 获取交通数据
    print("\n获取交通数据...")
    traffic_data = integration.get_traffic_data()
    if traffic_data:
        print(f"数据源: {traffic_data.get('data_source', 'unknown')}")
        print(f"路段数: {traffic_data.get('total_segments', 0)}")
        print(f"质量评分: {traffic_data.get('quality_metrics', {}).get('overall_score', 0):.2%}")
    
    # 获取天气数据
    print("\n获取天气数据...")
    weather_data = integration.get_weather_data()
    if weather_data:
        print(f"天气: {weather_data.get('condition', 'unknown')}")
        print(f"温度: {weather_data.get('temperature', 0)}°C")
    
    # 获取性能报告
    print("\n性能报告:")
    perf_report = integration.get_performance_report(1)
    print(json.dumps(perf_report, indent=2, ensure_ascii=False))
    
    # 获取质量报告
    print("\n质量报告:")
    quality_report = integration.get_data_quality_report(1)
    print(json.dumps(quality_report, indent=2, ensure_ascii=False))
    
    # 获取集成状态
    print("\n集成状态:")
    status = integration.get_status()
    print(f"运行状态: {status.is_running}")
    print(f"总请求数: {status.total_requests}")
    print(f"成功率: {status.successful_requests / max(status.total_requests, 1):.2%}")
    
    # 停止集成
    print("\n停止集成...")
    integration.stop_integration()
    
    print("真实数据集成演示完成")