"""
拥堵原因分析器
基于预测结果和历史数据进行多维度拥堵原因分析

主要功能：
1. 多维度原因分析（天气、事故、事件、高峰期等）
2. 因果关系推理和影响因子排序
3. 拥堵传播路径分析
4. 时间序列分析和趋势预测
5. 风险评估和预警机制
6. 拥堵源头识别和影响范围评估

作者：TrafficAI Team
日期：2025-11-05
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
import json
import time
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

# 导入现有的预测器
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.congestion_predictor import (
    CongestionPropagationPredictor, 
    RoadSegment, 
    PredictionResult,
    CongestionLevel,
    create_sample_data
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CauseType(Enum):
    """拥堵原因类型"""
    WEATHER = "weather"                    # 天气原因
    ACCIDENT = "accident"                  # 交通事故
    EVENT = "event"                        # 特殊事件
    PEAK_HOUR = "peak_hour"               # 高峰期
    ROADWORK = "roadwork"                 # 道路施工
    CAPACITY_LIMIT = "capacity_limit"     # 通行能力限制
    SIGNAL_CONTROL = "signal_control"     # 信号控制
    INCIDENT = "incident"                 # 突发事件
    UNKNOWN = "unknown"                   # 未知原因


class RiskLevel(Enum):
    """风险等级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class PropagationDirection(Enum):
    """传播方向"""
    UPSTREAM = "upstream"       # 上游
    DOWNSTREAM = "downstream"   # 下游
    BIDIRECTIONAL = "bidirectional"  # 双向
    LOCALIZED = "localized"     # 局部


@dataclass
class CongestionCause:
    """拥堵原因数据结构"""
    cause_id: str
    cause_type: CauseType
    location: Tuple[float, float]  # (经度, 纬度)
    severity: float  # 严重程度 (0-1)
    start_time: float
    end_time: Optional[float] = None
    affected_segments: List[str] = field(default_factory=list)
    impact_score: float = 0.0
    confidence: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelationship:
    """因果关系数据结构"""
    cause_id: str
    effect_id: str
    causal_strength: float  # 因果强度 (0-1)
    time_lag: float  # 时间延迟 (分钟)
    confidence: float
    relationship_type: str  # "direct", "indirect", "cascade"


@dataclass
class PropagationPath:
    """传播路径数据结构"""
    source_segment: str
    target_segments: List[str]
    direction: PropagationDirection
    propagation_speed: float  # km/h
    influence_range: float    # km
    attenuation_factor: float  # 衰减因子
    path_nodes: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """风险评估结果"""
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100
    risk_factors: Dict[str, float]
    predictions: Dict[str, Any]
    recommendations: List[str]
    alert_level: str
    timestamp: float


class WeatherAnalyzer:
    """天气分析器"""
    
    def __init__(self):
        self.weather_impact_weights = {
            'rain': 0.3,
            'snow': 0.5,
            'fog': 0.4,
            'wind': 0.2,
            'temperature': 0.1,
            'visibility': 0.3
        }
    
    def analyze_weather_impact(self, weather_data: Dict[str, Any], 
                             traffic_data: pd.DataFrame) -> Tuple[float, List[str]]:
        """
        分析天气对交通的影响
        
        Args:
            weather_data: 天气数据
            traffic_data: 交通数据
            
        Returns:
            (影响程度, 影响因子列表)
        """
        impact_score = 0.0
        impact_factors = []
        
        # 降雨影响
        if 'rain_intensity' in weather_data:
            rain_intensity = weather_data['rain_intensity']
            if rain_intensity > 0:
                rain_impact = min(rain_intensity / 50.0, 1.0) * self.weather_impact_weights['rain']
                impact_score += rain_impact
                impact_factors.append(f"降雨强度{rain_intensity}mm/h")
        
        # 降雪影响
        if 'snow_intensity' in weather_data:
            snow_intensity = weather_data['snow_intensity']
            if snow_intensity > 0:
                snow_impact = min(snow_intensity / 20.0, 1.0) * self.weather_impact_weights['snow']
                impact_score += snow_impact
                impact_factors.append(f"降雪强度{snow_intensity}cm/h")
        
        # 雾天影响
        if 'visibility' in weather_data:
            visibility = weather_data['visibility']
            if visibility < 1000:  # 能见度低于1km
                fog_impact = (1000 - visibility) / 1000 * self.weather_impact_weights['fog']
                impact_score += fog_impact
                impact_factors.append(f"能见度{visibility}m")
        
        # 强风影响
        if 'wind_speed' in weather_data:
            wind_speed = weather_data['wind_speed']
            if wind_speed > 15:  # 大风
                wind_impact = min((wind_speed - 15) / 25.0, 1.0) * self.weather_impact_weights['wind']
                impact_score += wind_impact
                impact_factors.append(f"风速{wind_speed}m/s")
        
        # 极端温度影响
        if 'temperature' in weather_data:
            temp = weather_data['temperature']
            if temp < -10 or temp > 35:
                temp_impact = min(abs(temp - 20) / 25.0, 1.0) * self.weather_impact_weights['temperature']
                impact_score += temp_impact
                impact_factors.append(f"温度{temp}°C")
        
        # 归一化影响分数
        impact_score = min(impact_score, 1.0)
        
        return impact_score, impact_factors


class IncidentDetector:
    """事故检测器"""
    
    def __init__(self, anomaly_threshold: float = 0.8):
        self.anomaly_threshold = anomaly_threshold
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.traffic_patterns = {}
    
    def detect_incidents(self, traffic_data: pd.DataFrame, 
                        historical_data: pd.DataFrame) -> List[CongestionCause]:
        """
        检测交通事故
        
        Args:
            traffic_data: 当前交通数据
            historical_data: 历史交通数据
            
        Returns:
            检测到的事故列表
        """
        incidents = []
        
        # 计算交通指标异常
        anomalies = self._detect_traffic_anomalies(traffic_data, historical_data)
        
        # 识别事故模式
        for anomaly in anomalies:
            if anomaly['severity'] > self.anomaly_threshold:
                incident = CongestionCause(
                    cause_id=f"incident_{int(time.time())}_{anomaly['segment_id']}",
                    cause_type=CauseType.ACCIDENT,
                    location=anomaly['location'],
                    severity=anomaly['severity'],
                    start_time=time.time(),
                    affected_segments=[anomaly['segment_id']],
                    impact_score=anomaly['impact_score'],
                    confidence=anomaly['confidence'],
                    description=f"检测到路段{anomaly['segment_id']}交通事故",
                    metadata={'detection_method': 'anomaly_detection', 'anomaly_score': anomaly['score']}
                )
                incidents.append(incident)
        
        return incidents
    
    def _detect_traffic_anomalies(self, current_data: pd.DataFrame, 
                                 historical_data: pd.DataFrame) -> List[Dict]:
        """检测交通异常"""
        anomalies = []
        
        # 准备特征
        features = ['speed', 'flow', 'occupancy']
        
        # 计算历史统计
        historical_stats = historical_data.groupby('segment_id')[features].agg(['mean', 'std']).reset_index()
        
        for _, row in current_data.iterrows():
            segment_id = row['segment_id']
            
            # 获取该路段的历史统计
            hist_row = historical_stats[historical_stats['segment_id'] == segment_id]
            if hist_row.empty:
                continue
            
            hist_row = hist_row.iloc[0]
            
            # 计算Z-score异常
            anomaly_scores = {}
            for feature in features:
                current_value = row[feature]
                mean_val = hist_row[(feature, 'mean')]
                std_val = hist_row[(feature, 'std')]
                
                if std_val > 0:
                    z_score = abs((current_value - mean_val) / std_val)
                    anomaly_scores[feature] = z_score
            
            # 综合异常评分
            overall_score = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0
            
            if overall_score > 2.0:  # 2-sigma阈值
                severity = min(overall_score / 5.0, 1.0)
                impact_score = self._calculate_impact_score(row, anomaly_scores)
                
                anomalies.append({
                    'segment_id': segment_id,
                    'location': (row.get('longitude', 0), row.get('latitude', 0)),
                    'severity': severity,
                    'score': overall_score,
                    'impact_score': impact_score,
                    'confidence': min(overall_score / 3.0, 1.0),
                    'anomaly_scores': anomaly_scores
                })
        
        return anomalies
    
    def _calculate_impact_score(self, traffic_row: pd.Series, 
                               anomaly_scores: Dict[str, float]) -> float:
        """计算影响评分"""
        # 基于速度下降和占有率上升计算影响
        speed_impact = anomaly_scores.get('speed', 0) * 0.4
        flow_impact = anomaly_scores.get('flow', 0) * 0.3
        occupancy_impact = anomaly_scores.get('occupancy', 0) * 0.3
        
        return (speed_impact + flow_impact + occupancy_impact) / 3.0


class EventAnalyzer:
    """事件分析器"""
    
    def __init__(self):
        self.event_patterns = {
            'sports_event': {'time_pattern': [18, 19, 20], 'duration': 120},
            'concert': {'time_pattern': [19, 20, 21, 22], 'duration': 180},
            'conference': {'time_pattern': [8, 9, 17, 18], 'duration': 480},
            'festival': {'time_pattern': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'duration': 600}
        }
    
    def detect_events(self, traffic_data: pd.DataFrame, 
                     calendar_data: pd.DataFrame) -> List[CongestionCause]:
        """
        检测特殊事件
        
        Args:
            traffic_data: 交通数据
            calendar_data: 日历事件数据
            
        Returns:
            检测到的事件列表
        """
        events = []
        
        # 基于日历数据检测事件
        current_time = time.time()
        
        for _, event in calendar_data.iterrows():
            # 处理不同的日期时间格式
            try:
                event_start = pd.to_datetime(event['start_time']).timestamp()
                event_end = pd.to_datetime(event['end_time']).timestamp()
            except:
                # 如果转换失败，使用当前时间作为默认值
                event_start = current_time - 1800  # 30分钟前
                event_end = current_time + 3600   # 1小时后
            
            # 扩展时间窗口：当前时间前后4小时内的事件都考虑
            time_window = 14400  # 4小时
            
            if (event_start - time_window) <= current_time <= (event_end + time_window):
                # 检查事件类型和影响
                event_type = event.get('type', 'general')
                location = (event.get('longitude', 0), event.get('latitude', 0))
                
                # 计算事件影响
                impact_score = self._calculate_event_impact(event, traffic_data)
                
                event_cause = CongestionCause(
                    cause_id=f"event_{event['id']}",
                    cause_type=CauseType.EVENT,
                    location=location,
                    severity=impact_score,
                    start_time=event_start,
                    end_time=event_end,
                    affected_segments=self._find_affected_segments(location, traffic_data),
                    impact_score=impact_score,
                    confidence=0.8,
                    description=f"检测到{event_type}事件: {event.get('name', 'Unknown')}",
                    metadata={'event_type': event_type, 'expected_attendance': event.get('attendance', 0)}
                )
                events.append(event_cause)
        
        return events
    
    def _calculate_event_impact(self, event: pd.Series, 
                               traffic_data: pd.DataFrame) -> float:
        """计算事件影响"""
        # 基于参与人数和事件类型计算影响
        attendance = event.get('attendance', 0)
        event_type = event.get('type', 'general')
        
        # 基础影响评分
        base_impact = min(attendance / 10000.0, 1.0) * 0.6
        
        # 事件类型调整
        type_multipliers = {
            'sports_event': 1.2,
            'concert': 1.1,
            'conference': 0.8,
            'festival': 1.3
        }
        
        multiplier = type_multipliers.get(event_type, 1.0)
        impact_score = base_impact * multiplier
        
        return min(impact_score, 1.0)
    
    def _find_affected_segments(self, event_location: Tuple[float, float], 
                               traffic_data: pd.DataFrame) -> List[str]:
        """找到受影响的路段"""
        affected_segments = []
        
        # 简化实现：假设事件影响半径为2km
        for _, row in traffic_data.iterrows():
            seg_location = (row.get('longitude', 0), row.get('latitude', 0))
            distance = np.sqrt((event_location[0] - seg_location[0])**2 + 
                             (event_location[1] - seg_location[1])**2)
            
            if distance < 0.02:  # 约2km
                affected_segments.append(row['segment_id'])
        
        return affected_segments


class CausalInferenceEngine:
    """因果推理引擎"""
    
    def __init__(self):
        self.causal_graph = defaultdict(list)
        self.correlation_matrix = None
        self.mutual_information_matrix = None
    
    def build_causal_graph(self, data: pd.DataFrame, 
                          causes: List[CongestionCause]) -> List[CausalRelationship]:
        """
        构建因果图
        
        Args:
            data: 交通数据
            causes: 拥堵原因列表
            
        Returns:
            因果关系列表
        """
        relationships = []
        
        # 计算变量间的相关性
        features = ['speed', 'flow', 'occupancy', 'v_c_ratio']
        correlation_matrix = data[features].corr()
        
        # 计算互信息
        mi_matrix = self._calculate_mutual_information(data[features])
        
        # 构建因果关系
        for cause in causes:
            cause_effects = self._find_cause_effects(cause, data, correlation_matrix, mi_matrix)
            relationships.extend(cause_effects)
        
        return relationships
    
    def _calculate_mutual_information(self, data: pd.DataFrame) -> np.ndarray:
        """计算互信息矩阵"""
        n_features = data.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    # 离散化连续变量
                    x_discrete = pd.cut(data.iloc[:, i], bins=10, labels=False)
                    y_discrete = pd.cut(data.iloc[:, j], bins=10, labels=False)
                    
                    mi_matrix[i, j] = mutual_info_score(x_discrete, y_discrete)
        
        return mi_matrix
    
    def _find_cause_effects(self, cause: CongestionCause, 
                           data: pd.DataFrame, 
                           correlation_matrix: pd.DataFrame,
                           mi_matrix: np.ndarray) -> List[CausalRelationship]:
        """找到原因的影响效应"""
        effects = []
        
        # 基于空间邻近性查找可能的影响路段
        for segment_id in cause.affected_segments:
            # 查找邻近路段
            nearby_segments = self._find_nearby_segments(segment_id, data)
            
            for target_segment in nearby_segments:
                # 计算因果强度
                causal_strength = self._calculate_causal_strength(
                    cause, segment_id, target_segment, correlation_matrix, mi_matrix
                )
                
                if causal_strength > 0.3:  # 阈值
                    relationship = CausalRelationship(
                        cause_id=cause.cause_id,
                        effect_id=target_segment,
                        causal_strength=causal_strength,
                        time_lag=self._estimate_time_lag(cause, segment_id, target_segment),
                        confidence=cause.confidence * causal_strength,
                        relationship_type="direct"
                    )
                    effects.append(relationship)
        
        return effects
    
    def _find_nearby_segments(self, source_segment: str, data: pd.DataFrame) -> List[str]:
        """找到邻近路段"""
        # 简化实现：返回所有其他路段
        return [seg for seg in data['segment_id'].unique() if seg != source_segment]
    
    def _calculate_causal_strength(self, cause: CongestionCause, 
                                  source_segment: str, target_segment: str,
                                  correlation_matrix: pd.DataFrame,
                                  mi_matrix: np.ndarray) -> float:
        """计算因果强度"""
        # 基于相关性、互信息和原因严重程度计算
        spatial_correlation = 0.5  # 简化空间相关性
        
        # 原因类型权重
        type_weights = {
            CauseType.ACCIDENT: 0.9,
            CauseType.EVENT: 0.7,
            CauseType.WEATHER: 0.6,
            CauseType.PEAK_HOUR: 0.8,
            CauseType.ROADWORK: 0.5
        }
        
        type_weight = type_weights.get(cause.cause_type, 0.5)
        severity_weight = cause.severity
        
        causal_strength = spatial_correlation * type_weight * severity_weight
        
        return min(causal_strength, 1.0)
    
    def _estimate_time_lag(self, cause: CongestionCause, 
                          source_segment: str, target_segment: str) -> float:
        """估计时间延迟"""
        # 基于原因类型估计时间延迟
        lag_estimates = {
            CauseType.ACCIDENT: 5,      # 5分钟
            CauseType.EVENT: 15,        # 15分钟
            CauseType.WEATHER: 30,      # 30分钟
            CauseType.PEAK_HOUR: 60,    # 60分钟
            CauseType.ROADWORK: 120     # 120分钟
        }
        
        base_lag = lag_estimates.get(cause.cause_type, 30)
        
        # 根据严重程度调整
        adjusted_lag = base_lag * (1 - cause.severity * 0.5)
        
        return max(adjusted_lag, 1.0)


class PropagationAnalyzer:
    """传播分析器"""
    
    def __init__(self):
        self.propagation_models = {
            'wave': self._wave_propagation_model,
            'cascade': self._cascade_propagation_model,
            'diffusion': self._diffusion_propagation_model
        }
    
    def analyze_propagation_paths(self, causes: List[CongestionCause], 
                                traffic_network: Dict[str, Any]) -> List[PropagationPath]:
        """
        分析拥堵传播路径
        
        Args:
            causes: 拥堵原因列表
            traffic_network: 交通网络数据
            
        Returns:
            传播路径列表
        """
        propagation_paths = []
        
        for cause in causes:
            # 分析每个原因的传播路径
            paths = self._analyze_single_cause_propagation(cause, traffic_network)
            propagation_paths.extend(paths)
        
        return propagation_paths
    
    def _analyze_single_cause_propagation(self, cause: CongestionCause, 
                                        traffic_network: Dict[str, Any]) -> List[PropagationPath]:
        """分析单个原因的传播路径"""
        paths = []
        
        # 根据原因类型选择传播模型
        model_type = self._select_propagation_model(cause.cause_type)
        
        for source_segment in cause.affected_segments:
            # 计算传播路径
            propagation_result = self.propagation_models[model_type](
                source_segment, cause, traffic_network
            )
            
            if propagation_result:
                paths.append(propagation_result)
        
        return paths
    
    def _select_propagation_model(self, cause_type: CauseType) -> str:
        """选择传播模型"""
        model_mapping = {
            CauseType.ACCIDENT: 'wave',
            CauseType.EVENT: 'cascade',
            CauseType.WEATHER: 'diffusion',
            CauseType.PEAK_HOUR: 'wave',
            CauseType.ROADWORK: 'cascade'
        }
        
        return model_mapping.get(cause_type, 'wave')
    
    def _wave_propagation_model(self, source_segment: str, cause: CongestionCause, 
                              traffic_network: Dict[str, Any]) -> Optional[PropagationPath]:
        """波传播模型"""
        # 基于冲击波理论的传播模型
        propagation_speed = self._calculate_shockwave_speed(cause)
        influence_range = propagation_speed * 0.5  # 30分钟传播范围
        
        # 找到传播路径上的路段
        path_nodes = self._find_propagation_path(source_segment, influence_range, traffic_network)
        
        return PropagationPath(
            source_segment=source_segment,
            target_segments=path_nodes[1:],  # 排除源节点
            direction=PropagationDirection.DOWNSTREAM,
            propagation_speed=propagation_speed,
            influence_range=influence_range,
            attenuation_factor=0.8,
            path_nodes=path_nodes
        )
    
    def _cascade_propagation_model(self, source_segment: str, cause: CongestionCause, 
                                 traffic_network: Dict[str, Any]) -> Optional[PropagationPath]:
        """级联传播模型"""
        # 级联传播：基于网络拓扑的传播
        cascade_depth = int(cause.severity * 3)  # 级联深度
        path_nodes = self._find_cascade_path(source_segment, cascade_depth, traffic_network)
        
        return PropagationPath(
            source_segment=source_segment,
            target_segments=path_nodes[1:],
            direction=PropagationDirection.BIDIRECTIONAL,
            propagation_speed=5.0,  # 较低传播速度
            influence_range=len(path_nodes) * 1.0,  # 每节点1km
            attenuation_factor=0.6,
            path_nodes=path_nodes
        )
    
    def _diffusion_propagation_model(self, source_segment: str, cause: CongestionCause, 
                                   traffic_network: Dict[str, Any]) -> Optional[PropagationPath]:
        """扩散传播模型"""
        # 扩散传播：基于相似性的传播
        diffusion_radius = cause.severity * 5.0  # 扩散半径
        path_nodes = self._find_diffusion_path(source_segment, diffusion_radius, traffic_network)
        
        return PropagationPath(
            source_segment=source_segment,
            target_segments=path_nodes[1:],
            direction=PropagationDirection.LOCALIZED,
            propagation_speed=2.0,  # 很慢的传播速度
            influence_range=diffusion_radius,
            attenuation_factor=0.4,
            path_nodes=path_nodes
        )
    
    def _calculate_shockwave_speed(self, cause: CongestionCause) -> float:
        """计算冲击波速度"""
        # 基于原因严重程度和类型计算冲击波速度
        base_speeds = {
            CauseType.ACCIDENT: 15.0,
            CauseType.EVENT: 8.0,
            CauseType.WEATHER: 5.0,
            CauseType.PEAK_HOUR: 12.0,
            CauseType.ROADWORK: 6.0
        }
        
        base_speed = base_speeds.get(cause.cause_type, 10.0)
        adjusted_speed = base_speed * (1 - cause.severity * 0.3)
        
        return max(adjusted_speed, 1.0)
    
    def _find_propagation_path(self, source: str, max_distance: float, 
                             traffic_network: Dict[str, Any]) -> List[str]:
        """找到传播路径"""
        # 简化实现：返回线性路径
        return [f"{source}_upstream", source, f"{source}_downstream"]
    
    def _find_cascade_path(self, source: str, depth: int, 
                          traffic_network: Dict[str, Any]) -> List[str]:
        """找到级联路径"""
        path = [source]
        current = source
        
        for i in range(depth):
            next_node = f"{current}_cascade_{i+1}"
            path.append(next_node)
            current = next_node
        
        return path
    
    def _find_diffusion_path(self, source: str, radius: float, 
                           traffic_network: Dict[str, Any]) -> List[str]:
        """找到扩散路径"""
        # 基于半径找到周围节点
        return [source, f"{source}_nearby_1", f"{source}_nearby_2"]


class TimeSeriesAnalyzer:
    """时间序列分析器"""
    
    def __init__(self):
        self.trend_models = {}
        self.seasonal_patterns = {}
    
    def analyze_trends(self, time_series_data: pd.DataFrame, 
                      prediction_results: List[PredictionResult]) -> Dict[str, Any]:
        """
        分析时间序列趋势
        
        Args:
            time_series_data: 时间序列数据
            prediction_results: 预测结果
            
        Returns:
            趋势分析结果
        """
        analysis_results = {
            'trend_direction': {},
            'trend_strength': {},
            'seasonal_patterns': {},
            'anomaly_detection': {},
            'forecast_accuracy': {}
        }
        
        # 分析每个路段的趋势
        for segment_id in time_series_data['segment_id'].unique():
            segment_data = time_series_data[time_series_data['segment_id'] == segment_id]
            
            # 趋势分析
            trend_result = self._analyze_single_trend(segment_data)
            analysis_results['trend_direction'][segment_id] = trend_result['direction']
            analysis_results['trend_strength'][segment_id] = trend_result['strength']
            
            # 季节性模式分析
            seasonal_result = self._analyze_seasonal_patterns(segment_data)
            analysis_results['seasonal_patterns'][segment_id] = seasonal_result
            
            # 异常检测
            anomaly_result = self._detect_time_series_anomalies(segment_data)
            analysis_results['anomaly_detection'][segment_id] = anomaly_result
        
        # 预测准确性评估
        forecast_accuracy = self._evaluate_forecast_accuracy(prediction_results)
        analysis_results['forecast_accuracy'] = forecast_accuracy
        
        return analysis_results
    
    def _analyze_single_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析单个时间序列的趋势"""
        if len(data) < 10:
            return {'direction': 'insufficient_data', 'strength': 0.0}
        
        # 计算速度趋势
        speeds = data['speed'].values
        time_points = np.arange(len(speeds))
        
        # 线性回归分析趋势
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, speeds)
        
        # 趋势方向和强度
        if abs(slope) < 0.1:
            direction = 'stable'
            strength = 0.0
        elif slope > 0:
            direction = 'improving'
            strength = min(abs(slope) / 2.0, 1.0)
        else:
            direction = 'deteriorating'
            strength = min(abs(slope) / 2.0, 1.0)
        
        return {
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value
        }
    
    def _analyze_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析季节性模式"""
        # 提取时间特征
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        
        # 计算每小时平均速度
        hourly_pattern = data.groupby('hour')['speed'].mean()
        
        # 计算工作日vs周末模式
        weekday_pattern = data[data['day_of_week'] < 5]['speed'].mean()
        weekend_pattern = data[data['day_of_week'] >= 5]['speed'].mean()
        
        # 识别高峰时段
        peak_hours = hourly_pattern.nlargest(3).index.tolist()
        
        return {
            'hourly_pattern': hourly_pattern.to_dict(),
            'peak_hours': peak_hours,
            'weekday_avg_speed': weekday_pattern,
            'weekend_avg_speed': weekend_pattern,
            'peak_to_offpeak_ratio': weekday_pattern / hourly_pattern.mean() if hourly_pattern.mean() > 0 else 1.0
        }
    
    def _detect_time_series_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检测时间序列异常"""
        if len(data) < 10:
            return {'anomalies': [], 'anomaly_rate': 0.0}
        
        # 使用孤立森林检测异常
        features = ['speed', 'flow', 'occupancy']
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        anomaly_labels = isolation_forest.fit_predict(data[features])
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        
        # 计算异常率
        anomaly_rate = len(anomaly_indices) / len(data)
        
        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_rate': anomaly_rate,
            'anomaly_timestamps': data.iloc[anomaly_indices]['timestamp'].tolist() if len(anomaly_indices) > 0 else []
        }
    
    def _evaluate_forecast_accuracy(self, prediction_results: List[PredictionResult]) -> Dict[str, float]:
        """评估预测准确性"""
        if not prediction_results:
            return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
        
        # 计算预测误差指标
        all_speeds = []
        all_predictions = []
        
        for result in prediction_results:
            all_speeds.extend(result.predicted_speeds)
            # 简化：使用置信度作为预测值
            all_predictions.extend(result.confidence_scores)
        
        # 计算误差指标
        mae = np.mean(np.abs(np.array(all_speeds) - np.array(all_predictions)))
        rmse = np.sqrt(np.mean((np.array(all_speeds) - np.array(all_predictions)) ** 2))
        mape = np.mean(np.abs((np.array(all_speeds) - np.array(all_predictions)) / np.array(all_speeds))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }


class RiskAssessmentEngine:
    """风险评估引擎"""
    
    def __init__(self):
        self.risk_weights = {
            'congestion_severity': 0.3,
            'propagation_risk': 0.25,
            'duration_risk': 0.2,
            'spatial_extent': 0.15,
            'impact_score': 0.1
        }
        
        self.alert_thresholds = {
            'low': 30,
            'medium': 50,
            'high': 70,
            'critical': 85
        }
    
    def assess_risk(self, causes: List[CongestionCause], 
                   propagation_paths: List[PropagationPath],
                   time_series_analysis: Dict[str, Any]) -> RiskAssessment:
        """
        综合风险评估
        
        Args:
            causes: 拥堵原因列表
            propagation_paths: 传播路径列表
            time_series_analysis: 时间序列分析结果
            
        Returns:
            风险评估结果
        """
        # 计算各项风险指标
        congestion_severity = self._calculate_congestion_severity(causes)
        propagation_risk = self._calculate_propagation_risk(propagation_paths)
        duration_risk = self._calculate_duration_risk(causes)
        spatial_extent = self._calculate_spatial_extent(causes)
        impact_score = self._calculate_impact_score(causes)
        
        # 综合风险评分
        risk_factors = {
            'congestion_severity': congestion_severity,
            'propagation_risk': propagation_risk,
            'duration_risk': duration_risk,
            'spatial_extent': spatial_extent,
            'impact_score': impact_score
        }
        
        overall_score = sum(
            risk_factors[factor] * self.risk_weights[factor] 
            for factor in risk_factors
        )
        
        # 确定风险等级
        risk_level = self._determine_risk_level(overall_score)
        
        # 生成预警和建议
        alert_level = self._determine_alert_level(overall_score)
        recommendations = self._generate_recommendations(risk_factors, causes)
        
        # 预测信息
        predictions = {
            'expected_duration': self._estimate_duration(causes),
            'affected_area': spatial_extent * 10,  # 假设每单位影响10km²
            'traffic_delay': overall_score * 15,   # 假设最大延误15分钟
            'economic_impact': overall_score * 10000  # 假设最大经济损失10万元
        }
        
        return RiskAssessment(
            overall_risk_level=risk_level,
            risk_score=overall_score,
            risk_factors=risk_factors,
            predictions=predictions,
            recommendations=recommendations,
            alert_level=alert_level,
            timestamp=time.time()
        )
    
    def _calculate_congestion_severity(self, causes: List[CongestionCause]) -> float:
        """计算拥堵严重程度"""
        if not causes:
            return 0.0
        
        # 基于原因严重程度和影响范围计算
        severity_scores = [cause.severity * cause.impact_score for cause in causes]
        return np.mean(severity_scores)
    
    def _calculate_propagation_risk(self, paths: List[PropagationPath]) -> float:
        """计算传播风险"""
        if not paths:
            return 0.0
        
        # 基于传播速度和影响范围计算
        propagation_risks = []
        for path in paths:
            risk = (path.propagation_speed / 20.0) * (path.influence_range / 10.0) * (1 - path.attenuation_factor)
            propagation_risks.append(min(risk, 1.0))
        
        return np.mean(propagation_risks)
    
    def _calculate_duration_risk(self, causes: List[CongestionCause]) -> float:
        """计算持续时间风险"""
        if not causes:
            return 0.0
        
        # 基于原因类型估计持续时间
        duration_estimates = {
            CauseType.ACCIDENT: 45,    # 45分钟
            CauseType.EVENT: 120,      # 2小时
            CauseType.WEATHER: 180,    # 3小时
            CauseType.PEAK_HOUR: 120,  # 2小时
            CauseType.ROADWORK: 480    # 8小时
        }
        
        total_duration = 0
        for cause in causes:
            estimated_duration = duration_estimates.get(cause.cause_type, 60)
            total_duration += estimated_duration * cause.severity
        
        # 归一化到0-1范围
        return min(total_duration / 480.0, 1.0)  # 最大8小时
    
    def _calculate_spatial_extent(self, causes: List[CongestionCause]) -> float:
        """计算空间影响范围"""
        if not causes:
            return 0.0
        
        # 基于受影响路段数量计算
        all_affected_segments = set()
        for cause in causes:
            if hasattr(cause, 'affected_segments') and cause.affected_segments:
                if isinstance(cause.affected_segments, list):
                    all_affected_segments.update(cause.affected_segments)
                elif isinstance(cause.affected_segments, str):
                    all_affected_segments.add(cause.affected_segments)
        
        # 假设最大影响范围为100个路段
        spatial_extent = len(all_affected_segments) / 100.0
        return min(spatial_extent, 1.0)
    
    def _calculate_impact_score(self, causes: List[CongestionCause]) -> float:
        """计算影响评分"""
        if not causes:
            return 0.0
        
        # 基于原因影响评分计算
        impact_scores = []
        for cause in causes:
            try:
                impact_score = float(cause.impact_score) * float(cause.confidence)
                impact_scores.append(impact_score)
            except (TypeError, ValueError):
                # 如果转换失败，使用默认值
                impact_scores.append(0.5)
        
        return np.mean(impact_scores) if impact_scores else 0.0
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """确定风险等级"""
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _determine_alert_level(self, risk_score: float) -> str:
        """确定预警等级"""
        # 确保风险评分在0-1范围内
        risk_score = max(0, min(1, risk_score))
        score = risk_score * 100  # 转换为0-100分数
        
        if score < self.alert_thresholds['low']:
            return 'green'  # 正常
        elif score < self.alert_thresholds['medium']:
            return 'yellow'  # 注意
        elif score < self.alert_thresholds['high']:
            return 'orange'  # 警告
        elif score < self.alert_thresholds['critical']:
            return 'red'     # 严重
        else:
            return 'purple'  # 紧急
    
    def _generate_recommendations(self, risk_factors: Dict[str, float], 
                                causes: List[CongestionCause]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于主要风险因子生成建议
        max_risk_factor = max(risk_factors.items(), key=lambda x: x[1])
        risk_type, risk_value = max_risk_factor
        
        if risk_type == 'congestion_severity' and risk_value > 0.6:
            recommendations.append("建议实施紧急交通疏导措施")
            recommendations.append("增加现场交通管理人员")
        
        elif risk_type == 'propagation_risk' and risk_value > 0.6:
            recommendations.append("启动上游路段预警机制")
            recommendations.append("准备替代路径诱导方案")
        
        elif risk_type == 'duration_risk' and risk_value > 0.6:
            recommendations.append("准备长期交通组织方案")
            recommendations.append("协调相关部门延长工作时间")
        
        # 基于原因类型生成具体建议
        for cause in causes:
            if cause.cause_type == CauseType.ACCIDENT:
                recommendations.append("立即通知交警部门处理事故")
                recommendations.append("准备事故现场清理方案")
            elif cause.cause_type == CauseType.WEATHER:
                recommendations.append("发布恶劣天气驾驶提醒")
                recommendations.append("准备除雪除冰设备")
            elif cause.cause_type == CauseType.EVENT:
                recommendations.append("协调活动主办方优化交通组织")
                recommendations.append("增加公共交通班次")
        
        # 通用建议
        recommendations.extend([
            "持续监控交通状况变化",
            "及时更新路况信息发布",
            "准备应急响应预案"
        ])
        
        return recommendations
    
    def _estimate_duration(self, causes: List[CongestionCause]) -> float:
        """估计拥堵持续时间"""
        if not causes:
            return 0.0
        
        # 基于原因类型和严重程度估计持续时间
        duration_estimates = {
            CauseType.ACCIDENT: 45,
            CauseType.EVENT: 120,
            CauseType.WEATHER: 180,
            CauseType.PEAK_HOUR: 120,
            CauseType.ROADWORK: 480
        }
        
        total_duration = 0
        for cause in causes:
            base_duration = duration_estimates.get(cause.cause_type, 60)
            adjusted_duration = base_duration * (1 + cause.severity * 0.5)
            total_duration += adjusted_duration
        
        return total_duration / len(causes)  # 平均持续时间


class CongestionAnalyzer:
    """拥堵原因分析器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化拥堵原因分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.predictor = CongestionPropagationPredictor(config)
        self.weather_analyzer = WeatherAnalyzer()
        self.incident_detector = IncidentDetector()
        self.event_analyzer = EventAnalyzer()
        self.causal_engine = CausalInferenceEngine()
        self.propagation_analyzer = PropagationAnalyzer()
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.risk_engine = RiskAssessmentEngine()
        
        # 数据存储
        self.historical_data = deque(maxlen=config.get('history_length', 1000))
        self.analysis_cache = {}
        
        # 分析参数
        self.analysis_window = config.get('analysis_window', 30)  # 分析时间窗口（分钟）
        self.correlation_threshold = config.get('correlation_threshold', 0.3)
        self.causality_threshold = config.get('causality_threshold', 0.5)
        
        logger.info("拥堵原因分析器初始化完成")
    
    def analyze_congestion_causes(self, current_data: Dict[str, Any], 
                                weather_data: Optional[Dict[str, Any]] = None,
                                calendar_data: Optional[pd.DataFrame] = None,
                                historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        综合分析拥堵原因
        
        Args:
            current_data: 当前交通数据
            weather_data: 天气数据
            calendar_data: 日历事件数据
            historical_data: 历史交通数据
            
        Returns:
            分析结果字典
        """
        logger.info("开始拥堵原因分析")
        
        # 1. 预测拥堵扩散
        prediction_results = self._predict_congestion_propagation(current_data)
        
        # 2. 多维度原因分析
        causes = self._analyze_multi_dimensional_causes(
            current_data, weather_data, calendar_data, historical_data
        )
        
        # 3. 因果关系推理
        causal_relationships = self._infer_causal_relationships(
            current_data, causes, historical_data
        )
        
        # 4. 传播路径分析
        propagation_paths = self._analyze_propagation_paths(
            causes, current_data
        )
        
        # 5. 时间序列分析
        time_series_analysis = self._analyze_time_series(
            current_data, prediction_results, historical_data
        )
        
        # 6. 风险评估
        risk_assessment = self.risk_engine.assess_risk(
            causes, propagation_paths, time_series_analysis
        )
        
        # 7. 生成综合报告
        analysis_report = self._generate_analysis_report(
            causes, causal_relationships, propagation_paths,
            time_series_analysis, risk_assessment, prediction_results
        )
        
        # 缓存分析结果
        self._cache_analysis_result(analysis_report)
        
        logger.info("拥堵原因分析完成")
        return analysis_report
    
    def _predict_congestion_propagation(self, current_data: Dict[str, Any]) -> List[PredictionResult]:
        """预测拥堵扩散"""
        # 构建路段数据
        segments = []
        for segment_id, data in current_data.items():
            segment = RoadSegment(
                segment_id=segment_id,
                length=data.get('length', 1.0),
                lanes=data.get('lanes', 3),
                capacity=data.get('capacity', 2000),
                free_flow_speed=data.get('free_flow_speed', 60),
                current_speed=data.get('current_speed', 50),
                current_flow=data.get('current_flow', 1500),
                occupancy=data.get('occupancy', 0.3)
            )
            segments.append(segment)
        
        # 执行预测
        prediction_results = self.predictor.predict_congestion_propagation(
            segments, prediction_horizon=6
        )
        
        return prediction_results
    
    def _analyze_multi_dimensional_causes(self, current_data: Dict[str, Any],
                                        weather_data: Optional[Dict[str, Any]],
                                        calendar_data: Optional[pd.DataFrame],
                                        historical_data: Optional[pd.DataFrame]) -> List[CongestionCause]:
        """多维度原因分析"""
        all_causes = []
        
        # 1. 天气原因分析
        if weather_data:
            weather_impact, weather_factors = self.weather_analyzer.analyze_weather_impact(
                weather_data, pd.DataFrame(current_data)
            )
            
            if weather_impact > 0.3:  # 天气影响阈值
                weather_cause = CongestionCause(
                    cause_id=f"weather_{int(time.time())}",
                    cause_type=CauseType.WEATHER,
                    location=(0, 0),  # 需要实际坐标
                    severity=weather_impact,
                    start_time=time.time(),
                    affected_segments=list(current_data.keys()),
                    impact_score=weather_impact,
                    confidence=0.8,
                    description=f"天气影响: {', '.join(weather_factors)}",
                    metadata={'weather_factors': weather_factors}
                )
                all_causes.append(weather_cause)
        
        # 2. 事故检测
        if historical_data is not None:
            incidents = self.incident_detector.detect_incidents(
                pd.DataFrame(current_data), historical_data
            )
            all_causes.extend(incidents)
        
        # 3. 事件检测
        if calendar_data is not None:
            events = self.event_analyzer.detect_events(
                pd.DataFrame(current_data), calendar_data
            )
            all_causes.extend(events)
        
        # 4. 高峰期检测
        peak_causes = self._detect_peak_hour_causes(current_data)
        all_causes.extend(peak_causes)
        
        # 5. 道路施工检测
        roadwork_causes = self._detect_roadwork_causes(current_data)
        all_causes.extend(roadwork_causes)
        
        return all_causes
    
    def _detect_peak_hour_causes(self, current_data: Dict[str, Any]) -> List[CongestionCause]:
        """检测高峰期原因"""
        current_hour = time.localtime(time.time()).tm_hour
        
        # 定义高峰时段
        morning_peak = (7, 9)
        evening_peak = (17, 19)
        
        peak_causes = []
        
        if (morning_peak[0] <= current_hour <= morning_peak[1] or 
            evening_peak[0] <= current_hour <= evening_peak[1]):
            
            # 计算高峰期影响程度
            congestion_level = self._calculate_congestion_level(current_data)
            
            if congestion_level > 0.6:  # 高峰期拥堵阈值
                peak_cause = CongestionCause(
                    cause_id=f"peak_hour_{int(time.time())}",
                    cause_type=CauseType.PEAK_HOUR,
                    location=(0, 0),
                    severity=congestion_level,
                    start_time=time.time(),
                    affected_segments=list(current_data.keys()),
                    impact_score=congestion_level * 0.8,
                    confidence=0.9,
                    description=f"高峰期拥堵，当前时间: {current_hour}:00",
                    metadata={'peak_type': 'morning' if current_hour < 12 else 'evening'}
                )
                peak_causes.append(peak_cause)
        
        return peak_causes
    
    def _detect_roadwork_causes(self, current_data: Dict[str, Any]) -> List[CongestionCause]:
        """检测道路施工原因"""
        # 简化实现：基于长期低速度和占有率检测施工
        roadwork_causes = []
        
        for segment_id, data in current_data.items():
            current_speed = data.get('current_speed', 50)
            free_flow_speed = data.get('free_flow_speed', 60)
            occupancy = data.get('occupancy', 0.3)
            
            # 施工特征：速度持续较低，但占有率不高
            speed_ratio = current_speed / free_flow_speed
            if speed_ratio < 0.5 and occupancy < 0.6:
                severity = (1 - speed_ratio) * 0.7
                
                roadwork_cause = CongestionCause(
                    cause_id=f"roadwork_{segment_id}_{int(time.time())}",
                    cause_type=CauseType.ROADWORK,
                    location=(0, 0),
                    severity=severity,
                    start_time=time.time(),
                    affected_segments=[segment_id],
                    impact_score=severity * 0.6,
                    confidence=0.7,
                    description=f"检测到路段{segment_id}可能存在道路施工",
                    metadata={'speed_ratio': speed_ratio, 'occupancy': occupancy}
                )
                roadwork_causes.append(roadwork_cause)
        
        return roadwork_causes
    
    def _calculate_congestion_level(self, current_data: Dict[str, Any]) -> float:
        """计算拥堵等级"""
        speed_ratios = []
        
        for data in current_data.values():
            current_speed = data.get('current_speed', 50)
            free_flow_speed = data.get('free_flow_speed', 60)
            speed_ratio = current_speed / free_flow_speed
            speed_ratios.append(speed_ratio)
        
        # 基于平均速度比计算拥堵等级
        avg_speed_ratio = np.mean(speed_ratios)
        congestion_level = 1 - avg_speed_ratio  # 拥堵程度 = 1 - 速度比
        
        return max(0, min(1, congestion_level))
    
    def _infer_causal_relationships(self, current_data: Dict[str, Any],
                                  causes: List[CongestionCause],
                                  historical_data: Optional[pd.DataFrame]) -> List[CausalRelationship]:
        """推理因果关系"""
        if not causes:
            return []
        
        # 构建因果图
        if historical_data is not None:
            causal_relationships = self.causal_engine.build_causal_graph(
                historical_data, causes
            )
        else:
            # 简化实现：基于空间邻近性推断因果关系
            causal_relationships = self._simplified_causal_inference(causes)
        
        return causal_relationships
    
    def _simplified_causal_inference(self, causes: List[CongestionCause]) -> List[CausalRelationship]:
        """简化的因果推理"""
        relationships = []
        
        for i, cause1 in enumerate(causes):
            for j, cause2 in enumerate(causes):
                if i != j:
                    # 基于原因类型和时间推断因果关系
                    causal_strength = self._calculate_simple_causal_strength(cause1, cause2)
                    
                    if causal_strength > self.causality_threshold:
                        relationship = CausalRelationship(
                            cause_id=cause1.cause_id,
                            effect_id=cause2.cause_id,
                            causal_strength=causal_strength,
                            time_lag=15.0,  # 假设15分钟延迟
                            confidence=cause1.confidence * cause2.confidence * causal_strength,
                            relationship_type="indirect"
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _calculate_simple_causal_strength(self, cause1: CongestionCause, 
                                        cause2: CongestionCause) -> float:
        """计算简单的因果强度"""
        # 基于原因类型组合计算因果强度
        type_combinations = {
            (CauseType.ACCIDENT, CauseType.PEAK_HOUR): 0.8,
            (CauseType.EVENT, CauseType.PEAK_HOUR): 0.7,
            (CauseType.WEATHER, CauseType.ACCIDENT): 0.6,
            (CauseType.ROADWORK, CauseType.PEAK_HOUR): 0.5
        }
        
        combination = (cause1.cause_type, cause2.cause_type)
        reverse_combination = (cause2.cause_type, cause1.cause_type)
        
        base_strength = type_combinations.get(combination, type_combinations.get(reverse_combination, 0.3))
        
        # 基于严重程度调整
        adjusted_strength = base_strength * (cause1.severity + cause2.severity) / 2
        
        return min(adjusted_strength, 1.0)
    
    def _analyze_propagation_paths(self, causes: List[CongestionCause],
                                 current_data: Dict[str, Any]) -> List[PropagationPath]:
        """分析传播路径"""
        if not causes:
            return []
        
        # 构建交通网络（简化）
        traffic_network = {
            'segments': list(current_data.keys()),
            'connections': self._build_simplified_network(current_data)
        }
        
        propagation_paths = self.propagation_analyzer.analyze_propagation_paths(
            causes, traffic_network
        )
        
        return propagation_paths
    
    def _build_simplified_network(self, current_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """构建简化的网络连接"""
        segments = list(current_data.keys())
        connections = {}
        
        # 假设线性连接
        for i, segment in enumerate(segments):
            neighbors = []
            if i > 0:
                neighbors.append(segments[i-1])
            if i < len(segments) - 1:
                neighbors.append(segments[i+1])
            connections[segment] = neighbors
        
        return connections
    
    def _analyze_time_series(self, current_data: Dict[str, Any],
                           prediction_results: List[PredictionResult],
                           historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """时间序列分析"""
        if historical_data is None:
            # 创建模拟时间序列数据
            time_series_data = self._create_mock_time_series_data(current_data)
        else:
            time_series_data = historical_data
        
        time_series_analysis = self.time_series_analyzer.analyze_trends(
            time_series_data, prediction_results
        )
        
        return time_series_analysis
    
    def _create_mock_time_series_data(self, current_data: Dict[str, Any]) -> pd.DataFrame:
        """创建模拟时间序列数据"""
        mock_data = []
        current_time = time.time()
        
        for segment_id, data in current_data.items():
            # 生成过去24小时的数据
            for hour_offset in range(24):
                timestamp = current_time - hour_offset * 3600
                
                # 添加时间变化模式
                hour = time.localtime(timestamp).tm_hour
                time_factor = 1.0
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰期
                    time_factor = 0.7
                elif 22 <= hour or hour <= 6:  # 夜间
                    time_factor = 1.2
                
                mock_data.append({
                    'segment_id': segment_id,
                    'timestamp': timestamp,
                    'speed': data.get('current_speed', 50) * time_factor * np.random.uniform(0.8, 1.2),
                    'flow': data.get('current_flow', 1500) * (2 - time_factor) * np.random.uniform(0.8, 1.2),
                    'occupancy': data.get('occupancy', 0.3) * (2 - time_factor) * np.random.uniform(0.8, 1.2)
                })
        
        return pd.DataFrame(mock_data)
    
    def _generate_analysis_report(self, causes: List[CongestionCause],
                                causal_relationships: List[CausalRelationship],
                                propagation_paths: List[PropagationPath],
                                time_series_analysis: Dict[str, Any],
                                risk_assessment: RiskAssessment,
                                prediction_results: List[PredictionResult]) -> Dict[str, Any]:
        """生成综合分析报告"""
        report = {
            'analysis_timestamp': time.time(),
            'summary': {
                'total_causes': len(causes),
                'primary_cause_type': self._get_primary_cause_type(causes),
                'overall_risk_level': risk_assessment.overall_risk_level.name,
                'risk_score': risk_assessment.risk_score,
                'alert_level': risk_assessment.alert_level
            },
            'causes': [
                {
                    'cause_id': cause.cause_id,
                    'cause_type': cause.cause_type.value,
                    'severity': cause.severity,
                    'impact_score': cause.impact_score,
                    'confidence': cause.confidence,
                    'description': cause.description,
                    'affected_segments': cause.affected_segments,
                    'metadata': cause.metadata
                }
                for cause in causes
            ],
            'causal_relationships': [
                {
                    'cause_id': rel.cause_id,
                    'effect_id': rel.effect_id,
                    'causal_strength': rel.causal_strength,
                    'time_lag': rel.time_lag,
                    'confidence': rel.confidence,
                    'relationship_type': rel.relationship_type
                }
                for rel in causal_relationships
            ],
            'propagation_paths': [
                {
                    'source_segment': path.source_segment,
                    'target_segments': path.target_segments,
                    'direction': path.direction.value,
                    'propagation_speed': path.propagation_speed,
                    'influence_range': path.influence_range,
                    'attenuation_factor': path.attenuation_factor,
                    'path_nodes': path.path_nodes
                }
                for path in propagation_paths
            ],
            'time_series_analysis': time_series_analysis,
            'risk_assessment': {
                'overall_risk_level': risk_assessment.overall_risk_level.name,
                'risk_score': risk_assessment.risk_score,
                'risk_factors': risk_assessment.risk_factors,
                'predictions': risk_assessment.predictions,
                'recommendations': risk_assessment.recommendations,
                'alert_level': risk_assessment.alert_level
            },
            'predictions': [
                {
                    'segment_id': pred.segment_id,
                    'predicted_speeds': pred.predicted_speeds.tolist(),
                    'predicted_flows': pred.predicted_flows.tolist(),
                    'congestion_levels': [level.name for level in pred.congestion_levels],
                    'confidence_scores': pred.confidence_scores.tolist(),
                    'propagation_speed': pred.propagation_speed,
                    'influence_range': pred.influence_range
                }
                for pred in prediction_results
            ]
        }
        
        return report
    
    def _get_primary_cause_type(self, causes: List[CongestionCause]) -> str:
        """获取主要原因类型"""
        if not causes:
            return "none"
        
        # 统计原因类型频次和严重程度
        type_scores = defaultdict(float)
        
        for cause in causes:
            type_scores[cause.cause_type.value] += cause.severity * cause.impact_score
        
        # 返回得分最高的原因类型
        primary_type = max(type_scores.items(), key=lambda x: x[1])
        return primary_type[0]
    
    def _cache_analysis_result(self, result: Dict[str, Any]):
        """缓存分析结果"""
        cache_key = f"analysis_{int(time.time() // 300)}"  # 5分钟缓存
        self.analysis_cache[cache_key] = result
        
        # 清理过期缓存
        if len(self.analysis_cache) > 100:
            oldest_key = min(self.analysis_cache.keys())
            del self.analysis_cache[oldest_key]
    
    def get_cached_analysis(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """获取缓存的分析结果"""
        cache_key = f"analysis_{int(timestamp // 300)}"
        return self.analysis_cache.get(cache_key)
    
    def export_analysis_report(self, report: Dict[str, Any], 
                             output_path: str, format: str = 'json'):
        """导出分析报告"""
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        elif format == 'csv':
            # 导出原因分析到CSV
            causes_df = pd.DataFrame(report['causes'])
            causes_df.to_csv(output_path.replace('.json', '_causes.csv'), 
                           index=False, encoding='utf-8')
        
        logger.info(f"分析报告已导出到: {output_path}")


def create_sample_congestion_analyzer() -> CongestionAnalyzer:
    """创建示例拥堵分析器"""
    config = {
        'input_dim': 4,
        'hidden_dim': 64,
        'output_dim': 3,
        'gcn_layers': 3,
        'lstm_layers': 2,
        'dropout': 0.1,
        'history_length': 1000,
        'analysis_window': 30,
        'correlation_threshold': 0.3,
        'causality_threshold': 0.5
    }
    
    return CongestionAnalyzer(config)


def visualize_analysis_results(report: Dict[str, Any], save_path: str = None):
    """可视化分析结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 原因类型分布
    cause_types = [cause['cause_type'] for cause in report['causes']]
    cause_counts = pd.Series(cause_types).value_counts()
    
    axes[0, 0].pie(cause_counts.values, labels=cause_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('拥堵原因类型分布')
    
    # 2. 风险因子分析
    risk_factors = report['risk_assessment']['risk_factors']
    factors = list(risk_factors.keys())
    scores = list(risk_factors.values())
    
    axes[0, 1].bar(factors, scores)
    axes[0, 1].set_title('风险因子分析')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 拥堵严重程度分布
    severities = [cause['severity'] for cause in report['causes']]
    axes[0, 2].hist(severities, bins=10, alpha=0.7)
    axes[0, 2].set_title('拥堵严重程度分布')
    axes[0, 2].set_xlabel('严重程度')
    axes[0, 2].set_ylabel('频次')
    
    # 4. 传播路径可视化（简化）
    propagation_paths = report['propagation_paths']
    if propagation_paths:
        path_lengths = [len(path['path_nodes']) for path in propagation_paths]
        axes[1, 0].bar(range(len(path_lengths)), path_lengths)
        axes[1, 0].set_title('传播路径长度分布')
        axes[1, 0].set_xlabel('路径索引')
        axes[1, 0].set_ylabel('路径长度')
    
    # 5. 预测准确性
    predictions = report['predictions']
    if predictions:
        avg_confidence = np.mean([np.mean(pred['confidence_scores']) for pred in predictions])
        axes[1, 1].bar(['平均置信度'], [avg_confidence])
        axes[1, 1].set_title('预测置信度')
        axes[1, 1].set_ylim(0, 1)
    
    # 6. 风险等级分布
    risk_score = report['risk_assessment']['risk_score']
    risk_level = report['risk_assessment']['overall_risk_level']
    
    axes[1, 2].text(0.5, 0.7, f'风险等级: {risk_level}', ha='center', va='center', 
                   transform=axes[1, 2].transAxes, fontsize=16, fontweight='bold')
    axes[1, 2].text(0.5, 0.3, f'风险评分: {risk_score:.2f}', ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=14)
    axes[1, 2].set_title('综合风险评估')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存到: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 示例用法
    logger.info("拥堵原因分析器演示")
    
    # 创建分析器
    analyzer = create_sample_congestion_analyzer()
    
    # 创建示例数据
    current_data = {
        f'segment_{i:03d}': {
            'length': np.random.uniform(0.5, 3.0),
            'lanes': np.random.randint(2, 6),
            'capacity': np.random.randint(1500, 3000),
            'free_flow_speed': np.random.uniform(50, 80),
            'current_speed': np.random.uniform(30, 70),
            'current_flow': np.random.randint(800, 2500),
            'occupancy': np.random.uniform(0.2, 0.8)
        }
        for i in range(20)
    }
    
    # 天气数据
    weather_data = {
        'rain_intensity': 15.0,
        'temperature': 5.0,
        'wind_speed': 8.0,
        'visibility': 800
    }
    
    # 日历事件数据
    calendar_data = pd.DataFrame({
        'id': ['event_001'],
        'name': ['体育比赛'],
        'type': ['sports_event'],
        'start_time': [pd.Timestamp.now()],
        'end_time': [pd.Timestamp.now() + pd.Timedelta(hours=2)],
        'attendance': [5000],
        'longitude': [116.397],
        'latitude': [39.908]
    })
    
    # 历史数据
    historical_data = pd.DataFrame({
        'segment_id': [f'segment_{i:03d}' for i in range(20) for _ in range(50)],
        'timestamp': [time.time() - i * 300 for i in range(50) for _ in range(20)],
        'speed': np.random.uniform(30, 80, 1000),
        'flow': np.random.uniform(500, 3000, 1000),
        'occupancy': np.random.uniform(0.1, 0.9, 1000)
    })
    
    # 执行分析
    logger.info("开始执行拥堵原因分析")
    analysis_result = analyzer.analyze_congestion_causes(
        current_data=current_data,
        weather_data=weather_data,
        calendar_data=calendar_data,
        historical_data=historical_data
    )
    
    # 输出分析结果摘要
    logger.info("=== 分析结果摘要 ===")
    logger.info(f"检测到原因数量: {analysis_result['summary']['total_causes']}")
    logger.info(f"主要原因类型: {analysis_result['summary']['primary_cause_type']}")
    logger.info(f"风险等级: {analysis_result['summary']['overall_risk_level']}")
    logger.info(f"风险评分: {analysis_result['summary']['risk_score']:.2f}")
    logger.info(f"预警等级: {analysis_result['summary']['alert_level']}")
    
    # 导出报告
    analyzer.export_analysis_report(
        analysis_result, 
        '/workspace/code/services/congestion_analysis_report.json'
    )
    
    # 可视化结果
    visualize_analysis_results(
        analysis_result, 
        '/workspace/code/services/analysis_visualization.png'
    )
    
    logger.info("拥堵原因分析器演示完成")