#!/usr/bin/env python3
"""
数据质量监控和验证模块
提供实时数据质量检查、异常检测和报告功能
"""

import os
import sys
import json
import time
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.append('/workspace/code')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"    # 95-100%
    GOOD = "good"             # 85-94%
    FAIR = "fair"             # 70-84%
    POOR = "poor"             # 50-69%
    CRITICAL = "critical"     # 0-49%

class AnomalyType(Enum):
    """异常类型"""
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    INCONSISTENT = "inconsistent"
    STALE_DATA = "stale_data"
    INVALID_RANGE = "invalid_range"
    DUPLICATE = "duplicate"

@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float        # 完整性 0-1
    accuracy: float           # 准确性 0-1
    timeliness: float         # 时效性 0-1
    consistency: float        # 一致性 0-1
    validity: float           # 有效性 0-1
    overall_score: float      # 总体评分 0-1
    quality_level: DataQualityLevel
    issues: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AnomalyAlert:
    """异常警报"""
    alert_id: str
    anomaly_type: AnomalyType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    location: Optional[str] = None
    segment_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False

class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, db_path: str = "/workspace/data/data_quality.db"):
        self.db_path = db_path
        self.alert_history = deque(maxlen=1000)
        self.quality_history = deque(maxlen=100)
        self.monitoring_active = False
        self.monitor_thread = None
        
        # 初始化数据库
        self._init_database()
        
        # 质量阈值配置
        self.quality_thresholds = {
            DataQualityLevel.EXCELLENT: 0.95,
            DataQualityLevel.GOOD: 0.85,
            DataQualityLevel.FAIR: 0.70,
            DataQualityLevel.POOR: 0.50
        }
        
        # 异常检测参数
        self.anomaly_params = {
            'speed_outlier_std': 3.0,      # 速度异常标准差倍数
            'flow_outlier_std': 3.0,       # 流量异常标准差倍数
            'max_data_age_minutes': 30,    # 最大数据年龄(分钟)
            'min_completeness': 0.80,      # 最小完整性要求
            'duplicate_threshold': 0.95    # 重复数据相似度阈值
        }
        
        logger.info("数据质量监控器初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建质量指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    segment_id TEXT,
                    completeness REAL,
                    accuracy REAL,
                    timeliness REAL,
                    consistency REAL,
                    validity REAL,
                    overall_score REAL,
                    quality_level TEXT,
                    issues TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建异常警报表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT,
                    anomaly_type TEXT,
                    severity TEXT,
                    description TEXT,
                    location TEXT,
                    segment_id TEXT,
                    timestamp TEXT,
                    resolved INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建数据源状态表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_source_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT,
                    status TEXT,
                    last_update TEXT,
                    response_time REAL,
                    error_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("数据质量监控数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据质量监控数据库初始化失败: {e}")
    
    def assess_data_quality(self, traffic_data: Dict[str, Any], weather_data: Dict[str, Any] = None) -> DataQualityMetrics:
        """评估数据质量"""
        try:
            issues = []
            
            # 1. 完整性评估
            completeness = self._assess_completeness(traffic_data)
            if completeness < self.anomaly_params['min_completeness']:
                issues.append(f"数据完整性不足: {completeness:.2%}")
            
            # 2. 准确性评估
            accuracy = self._assess_accuracy(traffic_data)
            if accuracy < 0.8:
                issues.append(f"数据准确性偏低: {accuracy:.2%}")
            
            # 3. 时效性评估
            timeliness = self._assess_timeliness(traffic_data)
            if timeliness < 0.7:
                issues.append(f"数据时效性不足: {timeliness:.2%}")
            
            # 4. 一致性评估
            consistency = self._assess_consistency(traffic_data)
            if consistency < 0.8:
                issues.append(f"数据一致性不足: {consistency:.2%}")
            
            # 5. 有效性评估
            validity = self._assess_validity(traffic_data)
            if validity < 0.9:
                issues.append(f"数据有效性不足: {validity:.2%}")
            
            # 计算总体评分
            overall_score = (completeness + accuracy + timeliness + consistency + validity) / 5
            
            # 确定质量等级
            quality_level = self._determine_quality_level(overall_score)
            
            # 创建质量指标对象
            metrics = DataQualityMetrics(
                completeness=completeness,
                accuracy=accuracy,
                timeliness=timeliness,
                consistency=consistency,
                validity=validity,
                overall_score=overall_score,
                quality_level=quality_level,
                issues=issues
            )
            
            # 保存到数据库
            self._save_quality_metrics(metrics, traffic_data)
            
            # 添加到历史记录
            self.quality_history.append(metrics)
            
            logger.info(f"数据质量评估完成，总体评分: {overall_score:.2%} ({quality_level.value})")
            return metrics
            
        except Exception as e:
            logger.error(f"数据质量评估失败: {e}")
            return DataQualityMetrics(
                completeness=0.0,
                accuracy=0.0,
                timeliness=0.0,
                consistency=0.0,
                validity=0.0,
                overall_score=0.0,
                quality_level=DataQualityLevel.CRITICAL,
                issues=[f"质量评估失败: {e}"]
            )
    
    def _assess_completeness(self, traffic_data: Dict[str, Any]) -> float:
        """评估数据完整性"""
        try:
            if 'segments' not in traffic_data or not traffic_data['segments']:
                return 0.0
            
            segments = traffic_data['segments']
            total_segments = len(segments)
            if total_segments == 0:
                return 0.0
            
            complete_segments = 0
            required_fields = ['id', 'lat', 'lng', 'speed', 'flow', 'density']
            
            for segment in segments:
                if all(field in segment and segment[field] is not None for field in required_fields):
                    complete_segments += 1
            
            completeness = complete_segments / total_segments
            return completeness
            
        except Exception as e:
            logger.error(f"评估数据完整性失败: {e}")
            return 0.0
    
    def _assess_accuracy(self, traffic_data: Dict[str, Any]) -> float:
        """评估数据准确性"""
        try:
            if 'segments' not in traffic_data:
                return 0.0
            
            segments = traffic_data['segments']
            if not segments:
                return 0.0
            
            accurate_segments = 0
            total_segments = len(segments)
            
            for segment in segments:
                # 检查数值合理性
                is_accurate = True
                
                # 速度合理性检查
                speed = segment.get('speed', 0)
                if speed < 0 or speed > 200:  # 超出合理范围
                    is_accurate = False
                
                # 流量合理性检查
                flow = segment.get('flow', 0)
                if flow < 0 or flow > 10000:  # 超出合理范围
                    is_accurate = False
                
                # 密度与流量一致性检查
                density = segment.get('density', 0)
                if density > 0 and flow / density > 200:  # 速度异常高
                    is_accurate = False
                
                if is_accurate:
                    accurate_segments += 1
            
            accuracy = accurate_segments / total_segments
            return accuracy
            
        except Exception as e:
            logger.error(f"评估数据准确性失败: {e}")
            return 0.0
    
    def _assess_timeliness(self, traffic_data: Dict[str, Any]) -> float:
        """评估数据时效性"""
        try:
            if 'timestamp' not in traffic_data:
                return 0.0
            
            try:
                timestamp = datetime.fromisoformat(traffic_data['timestamp'].replace('Z', '+00:00'))
                current_time = datetime.now()
                age_minutes = (current_time - timestamp.replace(tzinfo=None)).total_seconds() / 60
                
                # 根据数据年龄计算时效性
                if age_minutes <= 5:
                    timeliness = 1.0
                elif age_minutes <= 15:
                    timeliness = 0.9
                elif age_minutes <= 30:
                    timeliness = 0.7
                elif age_minutes <= 60:
                    timeliness = 0.5
                else:
                    timeliness = 0.0
                
                return timeliness
                
            except ValueError:
                logger.warning("时间戳格式无效")
                return 0.0
                
        except Exception as e:
            logger.error(f"评估数据时效性失败: {e}")
            return 0.0
    
    def _assess_consistency(self, traffic_data: Dict[str, Any]) -> float:
        """评估数据一致性"""
        try:
            if 'segments' not in traffic_data:
                return 0.0
            
            segments = traffic_data['segments']
            if len(segments) < 2:
                return 1.0  # 单个路段无法检查一致性
            
            # 检查相邻路段数据的一致性
            consistent_pairs = 0
            total_pairs = 0
            
            for i in range(len(segments) - 1):
                seg1 = segments[i]
                seg2 = segments[i + 1]
                
                # 检查位置一致性
                lat_diff = abs(seg1.get('lat', 0) - seg2.get('lat', 0))
                lng_diff = abs(seg1.get('lng', 0) - seg2.get('lng', 0))
                
                # 如果相邻路段距离过远，可能存在数据不一致
                if lat_diff > 0.1 or lng_diff > 0.1:
                    continue  # 跳过距离过远的路段
                
                total_pairs += 1
                
                # 检查速度一致性（相邻路段速度不应差异过大）
                speed1 = seg1.get('speed', 0)
                speed2 = seg2.get('speed', 0)
                speed_diff = abs(speed1 - speed2)
                
                if speed_diff <= 20:  # 速度差异在20km/h内认为一致
                    consistent_pairs += 1
            
            if total_pairs == 0:
                return 1.0  # 没有可比较的路段对
            
            consistency = consistent_pairs / total_pairs
            return consistency
            
        except Exception as e:
            logger.error(f"评估数据一致性失败: {e}")
            return 0.0
    
    def _assess_validity(self, traffic_data: Dict[str, Any]) -> float:
        """评估数据有效性"""
        try:
            if 'segments' not in traffic_data:
                return 0.0
            
            segments = traffic_data['segments']
            if not segments:
                return 0.0
            
            valid_segments = 0
            total_segments = len(segments)
            
            for segment in segments:
                is_valid = True
                
                # 检查必需字段
                required_fields = ['id', 'lat', 'lng', 'speed', 'flow']
                for field in required_fields:
                    if field not in segment or segment[field] is None:
                        is_valid = False
                        break
                
                # 检查数值类型
                if is_valid:
                    try:
                        lat = float(segment['lat'])
                        lng = float(segment['lng'])
                        speed = float(segment['speed'])
                        flow = int(segment['flow'])
                        
                        # 检查坐标范围
                        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                            is_valid = False
                        
                        # 检查数值范围
                        if speed < 0 or flow < 0:
                            is_valid = False
                            
                    except (ValueError, TypeError):
                        is_valid = False
                
                if is_valid:
                    valid_segments += 1
            
            validity = valid_segments / total_segments
            return validity
            
        except Exception as e:
            logger.error(f"评估数据有效性失败: {e}")
            return 0.0
    
    def _determine_quality_level(self, overall_score: float) -> DataQualityLevel:
        """确定数据质量等级"""
        if overall_score >= self.quality_thresholds[DataQualityLevel.EXCELLENT]:
            return DataQualityLevel.EXCELLENT
        elif overall_score >= self.quality_thresholds[DataQualityLevel.GOOD]:
            return DataQualityLevel.GOOD
        elif overall_score >= self.quality_thresholds[DataQualityLevel.FAIR]:
            return DataQualityLevel.FAIR
        elif overall_score >= self.quality_thresholds[DataQualityLevel.POOR]:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL
    
    def detect_anomalies(self, traffic_data: Dict[str, Any]) -> List[AnomalyAlert]:
        """检测数据异常"""
        try:
            alerts = []
            
            if 'segments' not in traffic_data:
                return alerts
            
            segments = traffic_data['segments']
            
            # 1. 检测缺失数据
            missing_alerts = self._detect_missing_data(segments)
            alerts.extend(missing_alerts)
            
            # 2. 检测异常值
            outlier_alerts = self._detect_outliers(segments)
            alerts.extend(outlier_alerts)
            
            # 3. 检测不一致数据
            inconsistent_alerts = self._detect_inconsistent_data(segments)
            alerts.extend(inconsistent_alerts)
            
            # 4. 检测过期数据
            stale_alerts = self._detect_stale_data(traffic_data)
            alerts.extend(stale_alerts)
            
            # 5. 检测重复数据
            duplicate_alerts = self._detect_duplicate_data(segments)
            alerts.extend(duplicate_alerts)
            
            # 保存警报到数据库
            for alert in alerts:
                self._save_anomaly_alert(alert)
                self.alert_history.append(alert)
            
            logger.info(f"检测到 {len(alerts)} 个数据异常")
            return alerts
            
        except Exception as e:
            logger.error(f"数据异常检测失败: {e}")
            return []
    
    def _detect_missing_data(self, segments: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """检测缺失数据"""
        alerts = []
        required_fields = ['speed', 'flow', 'density', 'lat', 'lng']
        
        for segment in segments:
            segment_id = segment.get('id', 'unknown')
            missing_fields = []
            
            for field in required_fields:
                if field not in segment or segment[field] is None:
                    missing_fields.append(field)
            
            if missing_fields:
                alert = AnomalyAlert(
                    alert_id=f"MISSING_{segment_id}_{int(time.time())}",
                    anomaly_type=AnomalyType.MISSING_DATA,
                    severity="medium" if len(missing_fields) <= 2 else "high",
                    description=f"路段 {segment_id} 缺失数据字段: {', '.join(missing_fields)}",
                    segment_id=segment_id
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_outliers(self, segments: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """检测异常值"""
        alerts = []
        
        try:
            # 提取速度数据用于异常检测
            speeds = [seg.get('speed', 0) for seg in segments if seg.get('speed') is not None]
            flows = [seg.get('flow', 0) for seg in segments if seg.get('flow') is not None]
            
            if len(speeds) < 3:  # 数据点太少，无法进行异常检测
                return alerts
            
            # 计算统计量
            speed_mean = np.mean(speeds)
            speed_std = np.std(speeds)
            flow_mean = np.mean(flows)
            flow_std = np.std(flows)
            
            # 检测速度异常
            speed_threshold = self.anomaly_params['speed_outlier_std'] * speed_std
            for segment in segments:
                segment_id = segment.get('id', 'unknown')
                speed = segment.get('speed', 0)
                
                if speed and abs(speed - speed_mean) > speed_threshold:
                    alert = AnomalyAlert(
                        alert_id=f"OUTLIER_SPEED_{segment_id}_{int(time.time())}",
                        anomaly_type=AnomalyType.OUTLIER,
                        severity="high" if abs(speed - speed_mean) > 2 * speed_threshold else "medium",
                        description=f"路段 {segment_id} 速度异常: {speed} km/h (正常范围: {speed_mean-speed_threshold:.1f}-{speed_mean+speed_threshold:.1f})",
                        segment_id=segment_id
                    )
                    alerts.append(alert)
            
            # 检测流量异常
            flow_threshold = self.anomaly_params['flow_outlier_std'] * flow_std
            for segment in segments:
                segment_id = segment.get('id', 'unknown')
                flow = segment.get('flow', 0)
                
                if flow and abs(flow - flow_mean) > flow_threshold:
                    alert = AnomalyAlert(
                        alert_id=f"OUTLIER_FLOW_{segment_id}_{int(time.time())}",
                        anomaly_type=AnomalyType.OUTLIER,
                        severity="medium",
                        description=f"路段 {segment_id} 流量异常: {flow} veh/h (正常范围: {flow_mean-flow_threshold:.0f}-{flow_mean+flow_threshold:.0f})",
                        segment_id=segment_id
                    )
                    alerts.append(alert)
            
        except Exception as e:
            logger.error(f"异常值检测失败: {e}")
        
        return alerts
    
    def _detect_inconsistent_data(self, segments: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """检测不一致数据"""
        alerts = []
        
        for segment in segments:
            segment_id = segment.get('id', 'unknown')
            speed = segment.get('speed', 0)
            flow = segment.get('flow', 0)
            density = segment.get('density', 0)
            
            # 检查速度与流量的逻辑一致性
            if speed > 0 and flow > 0:
                calculated_speed = flow / max(density, 1)
                if abs(speed - calculated_speed) > 30:  # 速度差异超过30km/h
                    alert = AnomalyAlert(
                        alert_id=f"INCONSISTENT_{segment_id}_{int(time.time())}",
                        anomaly_type=AnomalyType.INCONSISTENT,
                        severity="medium",
                        description=f"路段 {segment_id} 速度与流量不一致: 速度{speed}km/h, 流量{flow}veh/h, 密度{density}veh/km",
                        segment_id=segment_id
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _detect_stale_data(self, traffic_data: Dict[str, Any]) -> List[AnomalyAlert]:
        """检测过期数据"""
        alerts = []
        
        try:
            if 'timestamp' not in traffic_data:
                alert = AnomalyAlert(
                    alert_id=f"STALE_NO_TIMESTAMP_{int(time.time())}",
                    anomaly_type=AnomalyType.STALE_DATA,
                    severity="high",
                    description="数据缺少时间戳，无法判断时效性"
                )
                alerts.append(alert)
                return alerts
            
            timestamp = datetime.fromisoformat(traffic_data['timestamp'].replace('Z', '+00:00'))
            current_time = datetime.now()
            age_minutes = (current_time - timestamp.replace(tzinfo=None)).total_seconds() / 60
            
            if age_minutes > self.anomaly_params['max_data_age_minutes']:
                alert = AnomalyAlert(
                    alert_id=f"STALE_DATA_{int(time.time())}",
                    anomaly_type=AnomalyType.STALE_DATA,
                    severity="high" if age_minutes > 60 else "medium",
                    description=f"数据过期: {age_minutes:.1f}分钟前更新 (阈值: {self.anomaly_params['max_data_age_minutes']}分钟)",
                    location="全局"
                )
                alerts.append(alert)
        
        except Exception as e:
            logger.error(f"过期数据检测失败: {e}")
        
        return alerts
    
    def _detect_duplicate_data(self, segments: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """检测重复数据"""
        alerts = []
        
        try:
            # 按位置分组检测重复路段
            position_groups = defaultdict(list)
            
            for segment in segments:
                lat = round(segment.get('lat', 0), 4)  # 保留4位小数
                lng = round(segment.get('lng', 0), 4)
                position_key = f"{lat},{lng}"
                position_groups[position_key].append(segment)
            
            # 检查每个位置的路段数量
            for position, same_position_segments in position_groups.items():
                if len(same_position_segments) > 1:
                    segment_ids = [seg.get('id', 'unknown') for seg in same_position_segments]
                    alert = AnomalyAlert(
                        alert_id=f"DUPLICATE_{position}_{int(time.time())}",
                        anomaly_type=AnomalyType.DUPLICATE,
                        severity="low",
                        description=f"位置 {position} 存在重复路段: {', '.join(segment_ids)}",
                        location=position
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.error(f"重复数据检测失败: {e}")
        
        return alerts
    
    def _save_quality_metrics(self, metrics: DataQualityMetrics, traffic_data: Dict[str, Any]):
        """保存质量指标到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 保存全局质量指标
            cursor.execute('''
                INSERT INTO quality_metrics (
                    timestamp, segment_id, completeness, accuracy, timeliness, 
                    consistency, validity, overall_score, quality_level, issues
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp,
                'GLOBAL',
                metrics.completeness,
                metrics.accuracy,
                metrics.timeliness,
                metrics.consistency,
                metrics.validity,
                metrics.overall_score,
                metrics.quality_level.value,
                json.dumps(metrics.issues)
            ))
            
            # 保存每个路段的质量指标
            if 'segments' in traffic_data:
                for segment in traffic_data['segments']:
                    segment_id = segment.get('id', 'unknown')
                    
                    # 简单的路段质量评估
                    segment_completeness = 1.0 if all(field in segment for field in ['speed', 'flow', 'lat', 'lng']) else 0.5
                    segment_accuracy = 1.0 if 0 <= segment.get('speed', 0) <= 200 else 0.0
                    segment_timeliness = metrics.timeliness  # 继承全局时效性
                    segment_consistency = 1.0  # 简化处理
                    segment_validity = 1.0 if segment.get('lat') and segment.get('lng') else 0.0
                    segment_overall = (segment_completeness + segment_accuracy + segment_timeliness + segment_consistency + segment_validity) / 5
                    
                    cursor.execute('''
                        INSERT INTO quality_metrics (
                            timestamp, segment_id, completeness, accuracy, timeliness, 
                            consistency, validity, overall_score, quality_level, issues
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.timestamp,
                        segment_id,
                        segment_completeness,
                        segment_accuracy,
                        segment_timeliness,
                        segment_consistency,
                        segment_validity,
                        segment_overall,
                        self._determine_quality_level(segment_overall).value,
                        json.dumps([])
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存质量指标失败: {e}")
    
    def _save_anomaly_alert(self, alert: AnomalyAlert):
        """保存异常警报到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO anomaly_alerts (
                    alert_id, anomaly_type, severity, description, location, 
                    segment_id, timestamp, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.anomaly_type.value,
                alert.severity,
                alert.description,
                alert.location,
                alert.segment_id,
                alert.timestamp,
                1 if alert.resolved else 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存异常警报失败: {e}")
    
    def get_quality_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取质量报告"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 查询最近的质量指标
            query = '''
                SELECT timestamp, segment_id, overall_score, quality_level, issues
                FROM quality_metrics
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            '''.format(hours)
            
            df = pd.read_sql_query(query, conn)
            
            # 计算统计信息
            if not df.empty:
                avg_score = df['overall_score'].mean()
                quality_distribution = df['quality_level'].value_counts().to_dict()
                recent_issues = df['issues'].dropna().head(10).tolist()
            else:
                avg_score = 0.0
                quality_distribution = {}
                recent_issues = []
            
            # 查询最近的异常警报
            alert_query = '''
                SELECT anomaly_type, severity, COUNT(*) as count
                FROM anomaly_alerts
                WHERE timestamp >= datetime('now', '-{} hours')
                GROUP BY anomaly_type, severity
            '''.format(hours)
            
            alert_df = pd.read_sql_query(alert_query, conn)
            anomaly_summary = alert_df.to_dict('records') if not alert_df.empty else []
            
            conn.close()
            
            report = {
                'report_period': f'{hours}小时',
                'average_quality_score': round(avg_score, 3),
                'quality_distribution': quality_distribution,
                'recent_issues': recent_issues,
                'anomaly_summary': anomaly_summary,
                'total_data_points': len(df),
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def start_monitoring(self, interval_seconds: int = 300):
        """启动持续监控"""
        if self.monitoring_active:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,), daemon=True)
        self.monitor_thread.start()
        logger.info(f"数据质量监控已启动，监控间隔: {interval_seconds}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("数据质量监控已停止")
    
    def _monitoring_loop(self, interval_seconds: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 这里可以添加实际的监控逻辑
                # 例如：定期检查数据源状态、生成报告等
                logger.debug("执行数据质量监控检查...")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(interval_seconds)
    
    def generate_quality_charts(self, output_dir: str = "/workspace/charts"):
        """生成质量图表"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取最近24小时的数据
            report = self.get_quality_report(24)
            
            if 'error' in report:
                logger.error(f"无法生成图表: {report['error']}")
                return
            
            # 生成质量分布饼图
            if report['quality_distribution']:
                plt.figure(figsize=(10, 6))
                plt.pie(report['quality_distribution'].values(), 
                       labels=report['quality_distribution'].keys(),
                       autopct='%1.1f%%')
                plt.title('数据质量等级分布 (最近24小时)')
                plt.savefig(os.path.join(output_dir, 'quality_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 生成异常统计柱状图
            if report['anomaly_summary']:
                plt.figure(figsize=(12, 6))
                anomaly_types = [item['anomaly_type'] for item in report['anomaly_summary']]
                counts = [item['count'] for item in report['anomaly_summary']]
                
                plt.bar(anomaly_types, counts)
                plt.title('异常类型统计 (最近24小时)')
                plt.xlabel('异常类型')
                plt.ylabel('发生次数')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'anomaly_statistics.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"质量图表已生成到: {output_dir}")
            
        except Exception as e:
            logger.error(f"生成质量图表失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 初始化监控器
    monitor = DataQualityMonitor()
    
    # 模拟交通数据
    mock_traffic_data = {
        'timestamp': datetime.now().isoformat(),
        'segments': [
            {
                'id': 'SEG_001',
                'lat': 39.9042,
                'lng': 116.4074,
                'speed': 45.0,
                'flow': 1200,
                'density': 60,
                'occupancy': 30.0
            },
            {
                'id': 'SEG_002',
                'lat': 39.9050,
                'lng': 116.4080,
                'speed': 0,  # 异常值
                'flow': 0,
                'density': 0,
                'occupancy': 0.0
            }
        ]
    }
    
    # 评估数据质量
    print("评估数据质量...")
    quality_metrics = monitor.assess_data_quality(mock_traffic_data)
    print(f"总体评分: {quality_metrics.overall_score:.2%}")
    print(f"质量等级: {quality_metrics.quality_level.value}")
    print(f"问题: {quality_metrics.issues}")
    
    # 检测异常
    print("\n检测数据异常...")
    alerts = monitor.detect_anomalies(mock_traffic_data)
    for alert in alerts:
        print(f"异常: {alert.anomaly_type.value} - {alert.description}")
    
    # 生成质量报告
    print("\n生成质量报告...")
    report = monitor.get_quality_report(1)
    print(f"报告: {report}")
    
    # 生成图表
    print("\n生成质量图表...")
    monitor.generate_quality_charts()