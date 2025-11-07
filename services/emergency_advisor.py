"""
应急建议生成器
基于拥堵预测结果生成应急策略，支持多目标优化和实时调整

主要功能：
1. 基于拥堵预测结果的应急策略生成
2. 多类型应急建议（交通管制、路径引导、信号优化等）
3. 多目标优化（时间、距离、成本、安全性）
4. 实时建议更新和动态调整
5. 不同场景应急方案（事故、恶劣天气、大型活动等）
6. 建议效果评估和反馈机制

作者：TrafficAI Team
日期：2025-11-05
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import heapq

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmergencyType(Enum):
    """应急类型枚举"""
    TRAFFIC_CONTROL = "traffic_control"      # 交通管制
    ROUTE_GUIDANCE = "route_guidance"        # 路径引导
    SIGNAL_OPTIMIZATION = "signal_optimization"  # 信号优化
    CAPACITY_MANAGEMENT = "capacity_management"  # 容量管理
    INFORMATION_DISPLAY = "information_display"  # 信息发布
    RESOURCE_DEPLOYMENT = "resource_deployment"  # 资源部署


class EmergencyScenario(Enum):
    """应急场景枚举"""
    TRAFFIC_ACCIDENT = "traffic_accident"      # 交通事故
    BAD_WEATHER = "bad_weather"               # 恶劣天气
    LARGE_EVENT = "large_event"               # 大型活动
    ROADWORK = "roadwork"                     # 道路施工
    SPECIAL_VEHICLE = "special_vehicle"       # 特种车辆
    EMERGENCY_RESCUE = "emergency_rescue"     # 应急救援


class Priority(Enum):
    """优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RoadSegment:
    """道路路段数据类"""
    segment_id: str
    start_node: str
    end_node: str
    length: float  # 公里
    lanes: int
    current_speed: float  # km/h
    free_flow_speed: float  # km/h
    capacity: int  # 车辆/小时
    current_volume: int  # 当前流量
    congestion_level: int  # 0-3: 自由流到严重拥堵
    coordinates: Tuple[float, float] = field(default_factory=tuple)  # (lat, lon)


@dataclass
class TrafficLight:
    """交通信号灯数据类"""
    light_id: str
    location: Tuple[float, float]
    current_phase: str
    cycle_time: int  # 秒
    green_time: Dict[str, int]  # 各相位绿灯时间
    coordination_group: Optional[str] = None


@dataclass
class EmergencySuggestion:
    """应急建议数据类"""
    suggestion_id: str
    type: EmergencyType
    scenario: EmergencyScenario
    priority: Priority
    title: str
    description: str
    implementation_cost: float  # 实施成本
    expected_benefit: float  # 预期收益
    implementation_time: int  # 实施时间（分钟）
    affected_areas: List[str]  # 影响区域
    effectiveness_score: float  # 有效性评分 (0-1)
    confidence_level: float  # 信心度 (0-1)
    created_time: datetime = field(default_factory=datetime.now)
    estimated_duration: int = 0  # 预计持续时间（分钟）
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class OptimizationObjective:
    """优化目标数据类"""
    minimize_time: float = 1.0      # 时间权重
    minimize_distance: float = 0.8  # 距离权重
    minimize_cost: float = 0.6      # 成本权重
    maximize_safety: float = 1.2    # 安全性权重
    minimize_congestion: float = 1.0  # 拥堵减少权重


class EmergencyAdvisor:
    """应急建议生成器主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化应急建议生成器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        self.road_network: Dict[str, RoadSegment] = {}
        self.traffic_lights: Dict[str, TrafficLight] = {}
        self.active_suggestions: Dict[str, EmergencySuggestion] = {}
        self.suggestion_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.learning_data: Dict[str, Any] = {}
        
        # 优化目标权重
        self.optimization_objectives = OptimizationObjective()
        
        # 实时更新参数
        self.update_interval = 30  # 秒
        self.last_update_time = time.time()
        
        # 初始化建议模板
        self._initialize_suggestion_templates()
        
        logger.info("应急建议生成器初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "max_suggestions_per_scenario": 10,
            "min_effectiveness_threshold": 0.3,
            "max_implementation_cost": 100000,
            "update_frequency": 30,
            "learning_rate": 0.01,
            "confidence_threshold": 0.6
        }
    
    def _initialize_suggestion_templates(self):
        """初始化建议模板"""
        self.suggestion_templates = {
            EmergencyScenario.TRAFFIC_ACCIDENT: [
                {
                    "type": EmergencyType.TRAFFIC_CONTROL,
                    "title": "实施临时交通管制",
                    "description": "在事故路段实施临时交通管制，限制车辆通行",
                    "base_cost": 5000,
                    "base_benefit": 8000,
                    "implementation_time": 10
                },
                {
                    "type": EmergencyType.ROUTE_GUIDANCE,
                    "title": "启动绕行路径引导",
                    "description": "通过可变信息板和导航系统引导车辆绕行",
                    "base_cost": 2000,
                    "base_benefit": 12000,
                    "implementation_time": 5
                },
                {
                    "type": EmergencyType.SIGNAL_OPTIMIZATION,
                    "title": "优化信号配时",
                    "description": "调整事故点周边信号灯配时，疏导交通流",
                    "base_cost": 1000,
                    "base_benefit": 6000,
                    "implementation_time": 3
                }
            ],
            EmergencyScenario.BAD_WEATHER: [
                {
                    "type": EmergencyType.TRAFFIC_CONTROL,
                    "title": "实施限速管制",
                    "description": "根据天气情况实施临时限速措施",
                    "base_cost": 3000,
                    "base_benefit": 10000,
                    "implementation_time": 8
                },
                {
                    "type": EmergencyType.INFORMATION_DISPLAY,
                    "title": "发布天气预警信息",
                    "description": "通过多种渠道发布天气预警和驾驶建议",
                    "base_cost": 1000,
                    "base_benefit": 5000,
                    "implementation_time": 2
                },
                {
                    "type": EmergencyType.SIGNAL_OPTIMIZATION,
                    "title": "延长绿灯时间",
                    "description": "适当延长绿灯时间，确保车辆安全通过",
                    "base_cost": 500,
                    "base_benefit": 4000,
                    "implementation_time": 2
                }
            ],
            EmergencyScenario.LARGE_EVENT: [
                {
                    "type": EmergencyType.CAPACITY_MANAGEMENT,
                    "title": "增加临时车道",
                    "description": "临时增加车道或调整车道功能",
                    "base_cost": 15000,
                    "base_benefit": 25000,
                    "implementation_time": 60
                },
                {
                    "type": EmergencyType.ROUTE_GUIDANCE,
                    "title": "实施分时段引导",
                    "description": "对不同方向车辆实施分时段引导",
                    "base_cost": 8000,
                    "base_benefit": 20000,
                    "implementation_time": 20
                },
                {
                    "type": EmergencyType.SIGNAL_OPTIMIZATION,
                    "title": "事件专用信号配时",
                    "description": "为大型活动制定专用信号配时方案",
                    "base_cost": 3000,
                    "base_benefit": 15000,
                    "implementation_time": 30
                }
            ]
        }
    
    def load_road_network_data(self, road_data: Dict[str, Any]):
        """
        加载路网数据
        
        Args:
            road_data: 路网数据字典
        """
        logger.info("正在加载路网数据...")
        
        for segment_id, data in road_data.items():
            road_segment = RoadSegment(
                segment_id=segment_id,
                start_node=data.get('start_node', ''),
                end_node=data.get('end_node', ''),
                length=data.get('length', 0.0),
                lanes=data.get('lanes', 1),
                current_speed=data.get('current_speed', 0.0),
                free_flow_speed=data.get('free_flow_speed', 60.0),
                capacity=data.get('capacity', 1000),
                current_volume=data.get('current_volume', 0),
                congestion_level=data.get('congestion_level', 0),
                coordinates=tuple(data.get('coordinates', (0.0, 0.0)))
            )
            self.road_network[segment_id] = road_segment
        
        logger.info(f"成功加载 {len(self.road_network)} 个路段数据")
    
    def load_traffic_light_data(self, light_data: Dict[str, Any]):
        """
        加载交通信号灯数据
        
        Args:
            light_data: 信号灯数据字典
        """
        logger.info("正在加载交通信号灯数据...")
        
        for light_id, data in light_data.items():
            traffic_light = TrafficLight(
                light_id=light_id,
                location=tuple(data.get('location', (0.0, 0.0))),
                current_phase=data.get('current_phase', 'green'),
                cycle_time=data.get('cycle_time', 120),
                green_time=data.get('green_time', {'green': 60, 'yellow': 3, 'red': 57}),
                coordination_group=data.get('coordination_group')
            )
            self.traffic_lights[light_id] = traffic_light
        
        logger.info(f"成功加载 {len(self.traffic_lights)} 个信号灯数据")
    
    def analyze_congestion_prediction(self, prediction_data: Dict[str, Any]) -> List[EmergencySuggestion]:
        """
        基于拥堵预测结果分析并生成应急建议
        
        Args:
            prediction_data: 拥堵预测数据
            
        Returns:
            应急建议列表
        """
        logger.info("开始分析拥堵预测结果...")
        
        suggestions = []
        current_time = datetime.now()
        
        # 提取预测信息
        congestion_hotspots = prediction_data.get('hotspots', [])
        propagation_patterns = prediction_data.get('propagation_patterns', [])
        severity_levels = prediction_data.get('severity_levels', [])
        
        for i, hotspot in enumerate(congestion_hotspots):
            severity = severity_levels[i] if i < len(severity_levels) else 1
            affected_segments = hotspot.get('affected_segments', [])
            
            # 根据拥堵严重程度确定应急场景
            scenario = self._determine_emergency_scenario(severity, hotspot)
            
            # 生成针对性建议
            scenario_suggestions = self._generate_scenario_suggestions(
                hotspot, scenario, severity, affected_segments
            )
            
            suggestions.extend(scenario_suggestions)
        
        # 优化建议排序
        optimized_suggestions = self._optimize_suggestions(suggestions)
        
        # 记录建议
        for suggestion in optimized_suggestions:
            self.active_suggestions[suggestion.suggestion_id] = suggestion
            self.suggestion_history.append(suggestion)
        
        logger.info(f"生成了 {len(optimized_suggestions)} 条应急建议")
        return optimized_suggestions
    
    def _determine_emergency_scenario(self, severity: int, hotspot: Dict) -> EmergencyScenario:
        """确定应急场景类型"""
        # 基于拥堵严重程度和特征确定场景
        if severity >= 3:
            if hotspot.get('accident_indicators', False):
                return EmergencyScenario.TRAFFIC_ACCIDENT
            elif hotspot.get('weather_impact', False):
                return EmergencyScenario.BAD_WEATHER
            elif hotspot.get('event_related', False):
                return EmergencyScenario.LARGE_EVENT
        
        # 默认返回一般性拥堵场景
        return EmergencyScenario.TRAFFIC_ACCIDENT
    
    def _generate_scenario_suggestions(self, hotspot: Dict, scenario: EmergencyScenario, 
                                     severity: int, affected_segments: List[str]) -> List[EmergencySuggestion]:
        """生成特定场景的应急建议"""
        suggestions = []
        
        if scenario not in self.suggestion_templates:
            return suggestions
        
        templates = self.suggestion_templates[scenario]
        
        for i, template in enumerate(templates):
            # 根据严重程度调整建议参数
            severity_multiplier = 1 + (severity - 1) * 0.3
            
            suggestion = EmergencySuggestion(
                suggestion_id=f"{scenario.value}_{hotspot.get('id', 'unknown')}_{i}",
                type=template["type"],
                scenario=scenario,
                priority=self._calculate_priority(severity, template["type"]),
                title=template["title"],
                description=template["description"],
                implementation_cost=template["base_cost"] * severity_multiplier,
                expected_benefit=template["base_benefit"] * severity_multiplier,
                implementation_time=int(template["implementation_time"] * (2 - severity * 0.2)),
                affected_areas=affected_segments,
                effectiveness_score=self._calculate_effectiveness_score(severity, template["type"]),
                confidence_level=self._calculate_confidence_level(hotspot, severity),
                estimated_duration=int(30 + severity * 15),  # 30-75分钟
                resource_requirements=self._generate_resource_requirements(template["type"]),
                risk_factors=self._identify_risk_factors(scenario, severity)
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_priority(self, severity: int, emergency_type: EmergencyType) -> Priority:
        """计算建议优先级"""
        base_priority = severity
        
        # 根据应急类型调整优先级
        type_priority_map = {
            EmergencyType.TRAFFIC_CONTROL: 1,
            EmergencyType.ROUTE_GUIDANCE: 0,
            EmergencyType.SIGNAL_OPTIMIZATION: -1,
            EmergencyType.CAPACITY_MANAGEMENT: 2,
            EmergencyType.INFORMATION_DISPLAY: -2,
            EmergencyType.RESOURCE_DEPLOYMENT: 1
        }
        
        adjusted_priority = base_priority + type_priority_map.get(emergency_type, 0)
        
        if adjusted_priority >= 4:
            return Priority.CRITICAL
        elif adjusted_priority >= 3:
            return Priority.HIGH
        elif adjusted_priority >= 2:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _calculate_effectiveness_score(self, severity: int, emergency_type: EmergencyType) -> float:
        """计算建议有效性评分"""
        # 基础有效性评分
        base_scores = {
            EmergencyType.TRAFFIC_CONTROL: 0.8,
            EmergencyType.ROUTE_GUIDANCE: 0.9,
            EmergencyType.SIGNAL_OPTIMIZATION: 0.7,
            EmergencyType.CAPACITY_MANAGEMENT: 0.85,
            EmergencyType.INFORMATION_DISPLAY: 0.6,
            EmergencyType.RESOURCE_DEPLOYMENT: 0.75
        }
        
        base_score = base_scores.get(emergency_type, 0.5)
        
        # 根据严重程度调整
        severity_adjustment = (severity - 1) * 0.1
        
        # 添加随机因素模拟不确定性
        uncertainty = np.random.normal(0, 0.05)
        
        effectiveness = min(1.0, max(0.0, base_score + severity_adjustment + uncertainty))
        
        return effectiveness
    
    def _calculate_confidence_level(self, hotspot: Dict, severity: int) -> float:
        """计算建议信心度"""
        # 基于数据质量、预测准确性等因素计算信心度
        data_quality = hotspot.get('data_quality', 0.8)
        prediction_accuracy = hotspot.get('prediction_accuracy', 0.7)
        
        # 严重程度越高，信心度可能越低（因为情况更复杂）
        severity_factor = 1 - (severity - 1) * 0.1
        
        confidence = (data_quality * 0.4 + prediction_accuracy * 0.6) * severity_factor
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_resource_requirements(self, emergency_type: EmergencyType) -> Dict[str, Any]:
        """生成资源需求"""
        requirements = {
            EmergencyType.TRAFFIC_CONTROL: {
                "traffic_officers": 4,
                "barriers": 20,
                "signs": 10,
                "vehicles": 2
            },
            EmergencyType.ROUTE_GUIDANCE: {
                "variable_message_signs": 5,
                "traffic_officers": 2,
                "communication_equipment": 1
            },
            EmergencyType.SIGNAL_OPTIMIZATION: {
                "traffic_engineers": 1,
                "signal_technicians": 1,
                "testing_equipment": 1
            },
            EmergencyType.CAPACITY_MANAGEMENT: {
                "construction_crew": 8,
                "traffic_officers": 6,
                "equipment": ["cones", "barriers", "signs"]
            },
            EmergencyType.INFORMATION_DISPLAY: {
                "communication_officers": 2,
                "media_equipment": 1
            },
            EmergencyType.RESOURCE_DEPLOYMENT: {
                "emergency_vehicles": 3,
                "medical_staff": 2,
                "rescue_equipment": 1
            }
        }
        
        return requirements.get(emergency_type, {})
    
    def _identify_risk_factors(self, scenario: EmergencyScenario, severity: int) -> List[str]:
        """识别风险因素"""
        risk_factors = []
        
        # 基础风险因素
        if severity >= 3:
            risk_factors.append("高拥堵可能导致二次事故")
            risk_factors.append("应急响应时间可能延长")
        
        # 场景特定风险
        scenario_risks = {
            EmergencyScenario.TRAFFIC_ACCIDENT: [
                "事故现场安全风险",
                "救援车辆通行困难",
                "现场清理时间不确定"
            ],
            EmergencyScenario.BAD_WEATHER: [
                "天气条件持续恶化",
                "能见度降低",
                "路面湿滑风险"
            ],
            EmergencyScenario.LARGE_EVENT: [
                "人流量超预期",
                "停车位不足",
                "公共交通压力"
            ]
        }
        
        scenario_specific = scenario_risks.get(scenario, [])
        risk_factors.extend(scenario_specific)
        
        return risk_factors
    
    def _optimize_suggestions(self, suggestions: List[EmergencySuggestion]) -> List[EmergencySuggestion]:
        """优化建议排序"""
        if not suggestions:
            return []
        
        # 多目标优化
        def objective_function(suggestion):
            # 计算综合评分
            benefit_cost_ratio = suggestion.expected_benefit / max(suggestion.implementation_cost, 1)
            time_efficiency = 1 / max(suggestion.implementation_time, 1)
            priority_score = suggestion.priority.value / 4.0
            effectiveness = suggestion.effectiveness_score
            confidence = suggestion.confidence_level
            
            # 加权综合评分
            score = (
                benefit_cost_ratio * 0.3 +
                time_efficiency * 0.2 +
                priority_score * 0.25 +
                effectiveness * 0.15 +
                confidence * 0.1
            )
            
            return -score  # 最小化负值，即最大化正值
        
        # 排序建议
        optimized = sorted(suggestions, key=objective_function)
        
        # 限制建议数量
        max_suggestions = self.config.get('max_suggestions_per_scenario', 10)
        return optimized[:max_suggestions]
    
    def update_suggestions_realtime(self, new_prediction_data: Dict[str, Any]) -> List[EmergencySuggestion]:
        """
        实时更新建议
        
        Args:
            new_prediction_data: 新的预测数据
            
        Returns:
            更新的建议列表
        """
        current_time = time.time()
        
        # 检查是否需要更新
        if current_time - self.last_update_time < self.update_interval:
            return []
        
        logger.info("执行实时建议更新...")
        self.last_update_time = current_time
        
        # 清理过期的建议
        self._cleanup_expired_suggestions()
        
        # 生成新建议
        new_suggestions = self.analyze_congestion_prediction(new_prediction_data)
        
        # 评估现有建议效果
        self._evaluate_suggestion_performance()
        
        # 动态调整建议
        adjusted_suggestions = self._adjust_suggestions_based_on_feedback()
        
        return new_suggestions + adjusted_suggestions
    
    def _cleanup_expired_suggestions(self):
        """清理过期的建议"""
        current_time = datetime.now()
        expired_ids = []
        
        for suggestion_id, suggestion in self.active_suggestions.items():
            if suggestion.estimated_duration > 0:
                elapsed_time = (current_time - suggestion.created_time).total_seconds() / 60
                if elapsed_time > suggestion.estimated_duration:
                    expired_ids.append(suggestion_id)
        
        for suggestion_id in expired_ids:
            del self.active_suggestions[suggestion_id]
            logger.info(f"清理过期建议: {suggestion_id}")
    
    def _evaluate_suggestion_performance(self):
        """评估建议效果"""
        # 模拟效果评估（实际应用中需要真实数据）
        for suggestion_id, suggestion in self.active_suggestions.items():
            # 计算模拟效果指标
            simulated_effectiveness = self._simulate_effectiveness(suggestion)
            self.performance_metrics[f"effectiveness_{suggestion_id}"].append(simulated_effectiveness)
    
    def _simulate_effectiveness(self, suggestion: EmergencySuggestion) -> float:
        """模拟建议效果"""
        # 基于建议特征模拟效果
        base_effectiveness = suggestion.effectiveness_score
        
        # 添加时间衰减
        elapsed_minutes = (datetime.now() - suggestion.created_time).total_seconds() / 60
        time_decay = math.exp(-elapsed_minutes / 60)  # 1小时后效果衰减到37%
        
        # 添加随机变化
        random_factor = np.random.normal(1, 0.1)
        
        effectiveness = base_effectiveness * time_decay * random_factor
        return max(0.0, min(1.0, effectiveness))
    
    def _adjust_suggestions_based_on_feedback(self) -> List[EmergencySuggestion]:
        """基于反馈调整建议"""
        adjusted_suggestions = []
        
        # 分析历史效果数据
        for suggestion_id, suggestion in self.active_suggestions.items():
            performance_key = f"effectiveness_{suggestion_id}"
            if performance_key in self.performance_metrics:
                recent_performance = self.performance_metrics[performance_key][-5:]  # 最近5次记录
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    
                    # 如果效果不佳，考虑调整
                    if avg_performance < 0.5:
                        adjusted_suggestion = self._create_adjusted_suggestion(suggestion)
                        if adjusted_suggestion:
                            adjusted_suggestions.append(adjusted_suggestion)
        
        return adjusted_suggestions
    
    def _create_adjusted_suggestion(self, original: EmergencySuggestion) -> Optional[EmergencySuggestion]:
        """创建调整后的建议"""
        # 基于原始建议创建改进版本
        if original.type == EmergencyType.ROUTE_GUIDANCE:
            # 如果路径引导效果不佳，考虑交通管制
            adjusted = EmergencySuggestion(
                suggestion_id=f"{original.suggestion_id}_adjusted",
                type=EmergencyType.TRAFFIC_CONTROL,
                scenario=original.scenario,
                priority=original.priority,
                title=f"增强版: {original.title}",
                description=f"基于效果反馈的增强措施: {original.description}",
                implementation_cost=original.implementation_cost * 1.5,
                expected_benefit=original.expected_benefit * 1.3,
                implementation_time=original.implementation_time + 5,
                affected_areas=original.affected_areas,
                effectiveness_score=min(1.0, original.effectiveness_score * 1.2),
                confidence_level=original.confidence_level * 0.9,
                estimated_duration=original.estimated_duration + 10
            )
            return adjusted
        
        return None
    
    def get_optimization_recommendations(self, objectives: Optional[OptimizationObjective] = None) -> Dict[str, Any]:
        """
        获取优化建议
        
        Args:
            objectives: 优化目标权重
            
        Returns:
            优化建议字典
        """
        if objectives:
            self.optimization_objectives = objectives
        
        recommendations = {
            "current_performance": self._analyze_current_performance(),
            "optimization_suggestions": self._generate_optimization_suggestions(),
            "resource_allocation": self._optimize_resource_allocation(),
            "implementation_priorities": self._rank_implementation_priorities()
        }
        
        return recommendations
    
    def _analyze_current_performance(self) -> Dict[str, float]:
        """分析当前性能"""
        if not self.active_suggestions:
            return {"overall_effectiveness": 0.0}
        
        total_effectiveness = sum(s.effectiveness_score for s in self.active_suggestions.values())
        avg_effectiveness = total_effectiveness / len(self.active_suggestions)
        
        total_cost = sum(s.implementation_cost for s in self.active_suggestions.values())
        total_benefit = sum(s.expected_benefit for s in self.active_suggestions.values())
        benefit_cost_ratio = total_benefit / max(total_cost, 1)
        
        return {
            "overall_effectiveness": avg_effectiveness,
            "benefit_cost_ratio": benefit_cost_ratio,
            "active_suggestions_count": len(self.active_suggestions),
            "average_implementation_time": np.mean([s.implementation_time for s in self.active_suggestions.values()])
        }
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        performance = self._analyze_current_performance()
        
        # 检查是否有足够的建议进行分析
        if performance.get("active_suggestions_count", 0) == 0:
            return ["当前没有活跃的应急建议，建议先生成应急策略"]
        
        if performance.get("overall_effectiveness", 0) < 0.6:
            suggestions.append("建议调整建议生成策略，提高有效性评分")
        
        if performance.get("benefit_cost_ratio", 0) < 1.5:
            suggestions.append("建议优化成本效益比，考虑更经济的应急方案")
        
        if performance.get("average_implementation_time", 0) > 30:
            suggestions.append("建议优化实施流程，缩短应急响应时间")
        
        return suggestions
    
    def _optimize_resource_allocation(self) -> Dict[str, Any]:
        """优化资源分配"""
        resource_usage = defaultdict(float)
        
        for suggestion in self.active_suggestions.values():
            for resource, amount in suggestion.resource_requirements.items():
                if isinstance(amount, (int, float)):
                    resource_usage[resource] += amount
        
        total_cost = sum(s.implementation_cost for s in self.active_suggestions.values())
        
        return {
            "resource_distribution": dict(resource_usage),
            "cost_allocation": {
                "high_priority": sum(s.implementation_cost for s in self.active_suggestions.values() 
                                  if s.priority.value >= 3) / max(total_cost, 1),
                "medium_priority": sum(s.implementation_cost for s in self.active_suggestions.values() 
                                     if s.priority.value == 2) / max(total_cost, 1),
                "low_priority": sum(s.implementation_cost for s in self.active_suggestions.values() 
                                  if s.priority.value == 1) / max(total_cost, 1)
            }
        }
    
    def _rank_implementation_priorities(self) -> List[Dict[str, Any]]:
        """排名实施优先级"""
        ranked = []
        
        for suggestion in self.active_suggestions.values():
            priority_score = (
                suggestion.priority.value * 0.4 +
                suggestion.effectiveness_score * 0.3 +
                suggestion.confidence_level * 0.2 +
                (1 / max(suggestion.implementation_time, 1)) * 0.1
            )
            
            ranked.append({
                "suggestion_id": suggestion.suggestion_id,
                "title": suggestion.title,
                "priority_score": priority_score,
                "implementation_time": suggestion.implementation_time,
                "cost": suggestion.implementation_cost,
                "effectiveness": suggestion.effectiveness_score
            })
        
        return sorted(ranked, key=lambda x: x["priority_score"], reverse=True)
    
    def export_suggestions_report(self, format: str = "json") -> str:
        """
        导出建议报告
        
        Args:
            format: 导出格式 ("json" 或 "csv")
            
        Returns:
            报告内容字符串
        """
        if format == "json":
            return self._export_json_report()
        elif format == "csv":
            return self._export_csv_report()
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _export_json_report(self) -> str:
        """导出JSON格式报告"""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_suggestions": len(self.active_suggestions),
                "active_suggestions": len(self.active_suggestions),
                "average_effectiveness": np.mean([s.effectiveness_score for s in self.active_suggestions.values()]) if self.active_suggestions else 0,
                "total_cost": sum(s.implementation_cost for s in self.active_suggestions.values()),
                "total_expected_benefit": sum(s.expected_benefit for s in self.active_suggestions.values())
            },
            "suggestions": []
        }
        
        for suggestion in self.active_suggestions.values():
            suggestion_data = {
                "suggestion_id": suggestion.suggestion_id,
                "type": suggestion.type.value,
                "scenario": suggestion.scenario.value,
                "priority": suggestion.priority.name,
                "title": suggestion.title,
                "description": suggestion.description,
                "implementation_cost": suggestion.implementation_cost,
                "expected_benefit": suggestion.expected_benefit,
                "implementation_time": suggestion.implementation_time,
                "effectiveness_score": suggestion.effectiveness_score,
                "confidence_level": suggestion.confidence_level,
                "affected_areas": suggestion.affected_areas,
                "created_time": suggestion.created_time.isoformat(),
                "estimated_duration": suggestion.estimated_duration
            }
            report_data["suggestions"].append(suggestion_data)
        
        return json.dumps(report_data, ensure_ascii=False, indent=2)
    
    def _export_csv_report(self) -> str:
        """导出CSV格式报告"""
        if not self.active_suggestions:
            return "suggestion_id,type,scenario,priority,title,cost,benefit,effectiveness\n"
        
        lines = ["suggestion_id,type,scenario,priority,title,implementation_cost,expected_benefit,effectiveness_score,confidence_level"]
        
        for suggestion in self.active_suggestions.values():
            line = f"{suggestion.suggestion_id},{suggestion.type.value},{suggestion.scenario.value},{suggestion.priority.name},\"{suggestion.title}\",{suggestion.implementation_cost},{suggestion.expected_benefit},{suggestion.effectiveness_score:.3f},{suggestion.confidence_level:.3f}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_suggestions_count": len(self.active_suggestions),
            "suggestion_history_size": len(self.suggestion_history),
            "road_network_size": len(self.road_network),
            "traffic_lights_count": len(self.traffic_lights),
            "last_update_time": self.last_update_time,
            "update_interval": self.update_interval,
            "performance_metrics": dict(self.performance_metrics)
        }


def create_sample_road_network() -> Dict[str, Any]:
    """创建示例路网数据"""
    return {
        "segment_001": {
            "start_node": "node_A",
            "end_node": "node_B",
            "length": 2.5,
            "lanes": 3,
            "current_speed": 45.0,
            "free_flow_speed": 80.0,
            "capacity": 3600,
            "current_volume": 2800,
            "congestion_level": 2,
            "coordinates": (39.9042, 116.4074)
        },
        "segment_002": {
            "start_node": "node_B",
            "end_node": "node_C",
            "length": 1.8,
            "lanes": 2,
            "current_speed": 25.0,
            "free_flow_speed": 60.0,
            "capacity": 2400,
            "current_volume": 2200,
            "congestion_level": 3,
            "coordinates": (39.9052, 116.4084)
        },
        "segment_003": {
            "start_node": "node_C",
            "end_node": "node_D",
            "length": 3.2,
            "lanes": 4,
            "current_speed": 55.0,
            "free_flow_speed": 90.0,
            "capacity": 4800,
            "current_volume": 3200,
            "congestion_level": 1,
            "coordinates": (39.9062, 116.4094)
        }
    }


def create_sample_prediction_data() -> Dict[str, Any]:
    """创建示例预测数据"""
    return {
        "hotspots": [
            {
                "id": "hotspot_001",
                "location": (39.9052, 116.4084),
                "affected_segments": ["segment_001", "segment_002"],
                "severity": 3,
                "accident_indicators": True,
                "weather_impact": False,
                "event_related": False,
                "data_quality": 0.85,
                "prediction_accuracy": 0.78
            },
            {
                "id": "hotspot_002", 
                "location": (39.9062, 116.4094),
                "affected_segments": ["segment_003"],
                "severity": 2,
                "accident_indicators": False,
                "weather_impact": True,
                "event_related": False,
                "data_quality": 0.92,
                "prediction_accuracy": 0.82
            }
        ],
        "propagation_patterns": [
            {
                "from_segment": "segment_001",
                "to_segment": "segment_002",
                "propagation_probability": 0.75,
                "time_delay": 15
            }
        ],
        "severity_levels": [3, 2]
    }


if __name__ == "__main__":
    # 示例使用
    logger.info("应急建议生成器示例运行")
    
    # 创建应急建议生成器
    advisor = EmergencyAdvisor()
    
    # 加载示例数据
    road_network = create_sample_road_network()
    prediction_data = create_sample_prediction_data()
    
    advisor.load_road_network_data(road_network)
    
    # 生成应急建议
    suggestions = advisor.analyze_congestion_prediction(prediction_data)
    
    # 输出建议
    logger.info(f"生成了 {len(suggestions)} 条应急建议:")
    for suggestion in suggestions:
        logger.info(f"- {suggestion.title} (优先级: {suggestion.priority.name}, 有效性: {suggestion.effectiveness_score:.2f})")
    
    # 获取优化建议
    optimization = advisor.get_optimization_recommendations()
    logger.info(f"优化建议: {optimization['optimization_suggestions']}")
    
    # 导出报告
    json_report = advisor.export_suggestions_report("json")
    logger.info(f"JSON报告长度: {len(json_report)} 字符")
    
    # 获取系统状态
    status = advisor.get_system_status()
    logger.info(f"系统状态: {status}")