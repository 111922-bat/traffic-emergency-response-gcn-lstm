"""
动态路径规划器
考虑实时交通状况的智能路径规划系统
"""

import heapq
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
import logging
from enum import Enum
import json
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficEventType(Enum):
    """交通事件类型"""
    ACCIDENT = "事故"
    CONSTRUCTION = "施工"
    SPECIAL_EVENT = "特殊活动"
    WEATHER = "天气"
    CONGESTION = "拥堵"
    ROAD_CLOSURE = "道路封闭"


class TrafficSeverity(Enum):
    """交通事件严重程度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TrafficEvent:
    """交通事件数据类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: TrafficEventType = TrafficEventType.CONGESTION
    severity: TrafficSeverity = TrafficSeverity.LOW
    location: Tuple[float, float] = (0.0, 0.0)  # (latitude, longitude)
    affected_edges: Set[str] = field(default_factory=set)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    description: str = ""
    estimated_duration: int = 60  # 分钟
    
    def is_active(self, current_time: datetime = None) -> bool:
        """检查事件是否处于活跃状态"""
        if current_time is None:
            current_time = datetime.now()
        
        if self.end_time is None:
            # 如果没有结束时间，检查是否在预计持续时间内
            return current_time <= self.start_time + timedelta(minutes=self.estimated_duration)
        
        return self.start_time <= current_time <= self.end_time
    
    def get_impact_multiplier(self) -> float:
        """获取事件对交通的影响倍数"""
        severity_multipliers = {
            TrafficSeverity.LOW: 1.2,
            TrafficSeverity.MEDIUM: 1.5,
            TrafficSeverity.HIGH: 2.0,
            TrafficSeverity.CRITICAL: 3.0
        }
        return severity_multipliers[self.severity]


@dataclass
class TrafficData:
    """实时交通数据"""
    timestamp: datetime = field(default_factory=datetime.now)
    edge_id: str = ""
    current_speed: float = 50.0  # km/h
    free_flow_speed: float = 50.0  # km/h
    congestion_level: float = 0.0  # 0.0-1.0
    travel_time: float = 0.0  # 分钟
    incident_count: int = 0
    
    def update_congestion_level(self):
        """更新拥堵等级"""
        if self.free_flow_speed > 0:
            self.congestion_level = max(0.0, 1.0 - (self.current_speed / self.free_flow_speed))
    
    def get_weight_factor(self) -> float:
        """获取路径权重因子"""
        self.update_congestion_level()
        return 1.0 + (self.congestion_level * 2.0)  # 拥堵时权重增加


@dataclass
class PathNode:
    """路径节点"""
    node_id: str
    coordinates: Tuple[float, float]
    neighbors: Dict[str, float] = field(default_factory=dict)  # neighbor_id -> distance
    
    def add_neighbor(self, neighbor_id: str, distance: float):
        """添加邻居节点"""
        self.neighbors[neighbor_id] = distance


@dataclass
class PathSegment:
    """路径段"""
    start_node: str
    end_node: str
    distance: float
    estimated_time: float
    congestion_level: float = 0.0
    traffic_events: List[TrafficEvent] = field(default_factory=list)


@dataclass
class RoutePlan:
    """路径规划结果"""
    route_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    path: List[str] = field(default_factory=list)  # 节点ID列表
    segments: List[PathSegment] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    confidence_score: float = 1.0
    alternative_routes: List['RoutePlan'] = field(default_factory=list)


class TrafficPredictor:
    """交通状况预测器"""
    
    def __init__(self, prediction_horizon: int = 30):  # 预测30分钟
        self.prediction_horizon = prediction_horizon  # 分钟
        self.historical_data = defaultdict(list)
        self.pattern_cache = {}
        
    def add_historical_data(self, edge_id: str, traffic_data: TrafficData):
        """添加历史交通数据"""
        self.historical_data[edge_id].append(traffic_data)
        
        # 保持最近1000条记录
        if len(self.historical_data[edge_id]) > 1000:
            self.historical_data[edge_id] = self.historical_data[edge_id][-1000:]
    
    def predict_traffic(self, edge_id: str, target_time: datetime) -> Optional[TrafficData]:
        """预测特定时间的交通状况"""
        if edge_id not in self.historical_data:
            return None
        
        historical = self.historical_data[edge_id]
        if not historical:
            return None
        
        # 简单的基于历史数据的预测
        # 实际应用中可以使用更复杂的机器学习模型
        current_time = datetime.now()
        time_diff = (target_time - current_time).total_seconds() / 60  # 分钟
        
        if time_diff <= 0:
            # 返回最新的实际数据
            return historical[-1]
        
        # 基于历史模式预测
        predicted_data = TrafficData(
            edge_id=edge_id,
            timestamp=target_time
        )
        
        # 简单的趋势预测
        if len(historical) >= 2:
            recent_data = historical[-10:]  # 最近10条记录
            avg_speed = sum(d.current_speed for d in recent_data) / len(recent_data)
            avg_congestion = sum(d.congestion_level for d in recent_data) / len(recent_data)
            
            # 应用时间因子
            hour = target_time.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰期
                predicted_data.current_speed = avg_speed * 0.7
                predicted_data.congestion_level = min(1.0, avg_congestion * 1.3)
            elif 22 <= hour or hour <= 6:  # 夜间
                predicted_data.current_speed = avg_speed * 1.2
                predicted_data.congestion_level = max(0.0, avg_congestion * 0.5)
            else:
                predicted_data.current_speed = avg_speed
                predicted_data.congestion_level = avg_congestion
        
        predicted_data.update_congestion_level()
        return predicted_data


class DynamicPathPlanner:
    """动态路径规划器主类"""
    
    def __init__(self, 
                 stability_threshold: float = 0.1,
                 max_recalculation_interval: int = 300,  # 5分钟
                 prediction_horizon: int = 30):
        """
        初始化动态路径规划器
        
        Args:
            stability_threshold: 路径稳定性阈值
            max_recalculation_interval: 最大重新计算间隔（秒）
            prediction_horizon: 预测时间窗口（分钟）
        """
        self.nodes: Dict[str, PathNode] = {}
        self.traffic_data: Dict[str, TrafficData] = {}
        self.traffic_events: Dict[str, TrafficEvent] = {}
        self.active_routes: Dict[str, RoutePlan] = {}
        
        self.predictor = TrafficPredictor(prediction_horizon)
        self.stability_threshold = stability_threshold
        self.max_recalculation_interval = max_recalculation_interval
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'successful_recalculations': 0,
            'stability_violations': 0,
            'event_impacts': 0
        }
        
        logger.info("动态路径规划器初始化完成")
    
    def add_node(self, node_id: str, coordinates: Tuple[float, float]):
        """添加路径节点"""
        with self.lock:
            if node_id not in self.nodes:
                self.nodes[node_id] = PathNode(node_id, coordinates)
                logger.info(f"添加节点: {node_id}")
    
    def add_edge(self, start_node: str, end_node: str, distance: float):
        """添加路径边"""
        with self.lock:
            if start_node in self.nodes and end_node in self.nodes:
                self.nodes[start_node].add_neighbor(end_node, distance)
                self.nodes[end_node].add_neighbor(start_node, distance)
                logger.info(f"添加边: {start_node} -> {end_node}, 距离: {distance}km")
    
    def update_traffic_data(self, traffic_data: TrafficData):
        """更新实时交通数据"""
        with self.lock:
            self.traffic_data[traffic_data.edge_id] = traffic_data
            self.predictor.add_historical_data(traffic_data.edge_id, traffic_data)
            logger.debug(f"更新交通数据: {traffic_data.edge_id}, 速度: {traffic_data.current_speed}km/h")
    
    def add_traffic_event(self, event: TrafficEvent):
        """添加交通事件"""
        with self.lock:
            self.traffic_events[event.id] = event
            logger.info(f"添加交通事件: {event.event_type.value}, 位置: {event.location}")
    
    def remove_traffic_event(self, event_id: str):
        """移除交通事件"""
        with self.lock:
            if event_id in self.traffic_events:
                del self.traffic_events[event_id]
                logger.info(f"移除交通事件: {event_id}")
    
    def get_edge_weight(self, start_node: str, end_node: str, 
                       target_time: datetime = None) -> float:
        """获取边的权重（考虑交通状况）"""
        if target_time is None:
            target_time = datetime.now()
        
        edge_id = f"{start_node}_{end_node}"
        
        # 基础权重（距离）
        base_weight = self.nodes[start_node].neighbors.get(end_node, 1.0)
        
        # 交通状况调整
        traffic_multiplier = 1.0
        
        # 当前交通数据
        if edge_id in self.traffic_data:
            traffic_data = self.traffic_data[edge_id]
            traffic_multiplier *= traffic_data.get_weight_factor()
        
        # 预测交通数据
        predicted_data = self.predictor.predict_traffic(edge_id, target_time)
        if predicted_data:
            traffic_multiplier *= predicted_data.get_weight_factor()
        
        # 交通事件影响
        event_multiplier = 1.0
        current_time = datetime.now()
        
        for event in self.traffic_events.values():
            if (event.is_active(current_time) and 
                edge_id in event.affected_edges):
                event_multiplier *= event.get_impact_multiplier()
        
        return base_weight * traffic_multiplier * event_multiplier
    
    def heuristic(self, node_id: str, goal_id: str) -> float:
        """启发式函数（欧几里得距离）"""
        if node_id not in self.nodes or goal_id not in self.nodes:
            return 0.0
        
        node1 = self.nodes[node_id]
        node2 = self.nodes[goal_id]
        
        # 简化的欧几里得距离计算
        lat1, lon1 = node1.coordinates
        lat2, lon2 = node2.coordinates
        
        # 转换为公里（近似）
        distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # 1度约111km
        return distance
    
    def a_star(self, start: str, goal: str, target_time: datetime = None) -> Optional[RoutePlan]:
        """A*算法路径规划"""
        if target_time is None:
            target_time = datetime.now()
        
        if start not in self.nodes or goal not in self.nodes:
            return None
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {node: float('inf') for node in self.nodes}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.nodes}
        f_score[start] = self.heuristic(start, goal)
        
        visited = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal:
                # 重构路径
                path = [current]
                total_cost = g_score[current]
                
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                path.reverse()
                
                # 创建路径段
                segments = []
                total_distance = 0
                total_time = 0
                
                for i in range(len(path) - 1):
                    start_node = path[i]
                    end_node = path[i + 1]
                    
                    distance = self.nodes[start_node].neighbors.get(end_node, 0)
                    weight = self.get_edge_weight(start_node, end_node, target_time)
                    estimated_time = (distance / 50.0) * weight  # 假设50km/h基础速度
                    
                    segment = PathSegment(
                        start_node=start_node,
                        end_node=end_node,
                        distance=distance,
                        estimated_time=estimated_time,
                        congestion_level=self.traffic_data.get(f"{start_node}_{end_node}", TrafficData()).congestion_level
                    )
                    
                    segments.append(segment)
                    total_distance += distance
                    total_time += estimated_time
                
                route = RoutePlan(
                    path=path,
                    segments=segments,
                    total_distance=total_distance,
                    total_time=total_time,
                    confidence_score=self._calculate_confidence(segments)
                )
                
                return route
            
            for neighbor in self.nodes[current].neighbors:
                if neighbor in visited:
                    continue
                
                # 计算移动成本
                weight = self.get_edge_weight(current, neighbor, target_time)
                tentative_g_score = g_score[current] + weight
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _calculate_confidence(self, segments: List[PathSegment]) -> float:
        """计算路径置信度"""
        if not segments:
            return 0.0
        
        # 基于拥堵程度和事件影响计算置信度
        total_congestion = sum(seg.congestion_level for seg in segments)
        avg_congestion = total_congestion / len(segments)
        
        confidence = max(0.1, 1.0 - avg_congestion * 0.8)
        return confidence
    
    def plan_route(self, start: str, goal: str, 
                   time_windows: List[Tuple[datetime, datetime]] = None) -> RoutePlan:
        """规划路径（支持多时间窗口）"""
        with self.lock:
            self.stats['total_calculations'] += 1
            
            if time_windows is None:
                # 默认时间窗口：当前时间到未来30分钟
                current_time = datetime.now()
                time_windows = [(current_time, current_time + timedelta(minutes=30))]
            
            best_route = None
            best_score = float('inf')
            
            for start_time, end_time in time_windows:
                # 在时间窗口内寻找最优路径
                route = self.a_star(start, goal, start_time)
                
                if route:
                    # 计算时间窗口内的路径评分
                    score = self._evaluate_route_in_time_window(route, start_time, end_time)
                    
                    if score < best_score:
                        best_score = score
                        best_route = route
            
            if best_route:
                # 生成备选路径
                best_route.alternative_routes = self._generate_alternative_routes(start, goal, time_windows[0][0])
                self.stats['successful_recalculations'] += 1
                logger.info(f"路径规划成功: {start} -> {goal}, 距离: {best_route.total_distance:.2f}km")
            else:
                logger.warning(f"路径规划失败: {start} -> {goal}")
            
            return best_route
    
    def _evaluate_route_in_time_window(self, route: RoutePlan, 
                                     start_time: datetime, end_time: datetime) -> float:
        """评估路径在时间窗口内的表现"""
        # 考虑拥堵、事件和时间成本
        congestion_penalty = sum(seg.congestion_level for seg in route.segments) * 10
        event_penalty = len([seg for seg in route.segments if seg.traffic_events]) * 20
        time_penalty = max(0, route.total_time - 30) * 0.5  # 超过30分钟的惩罚
        
        return congestion_penalty + event_penalty + time_penalty
    
    def _generate_alternative_routes(self, start: str, goal: str, 
                                   target_time: datetime, max_alternatives: int = 3) -> List[RoutePlan]:
        """生成备选路径"""
        alternatives = []
        
        # 简单的备选路径生成：移除一些边或节点
        for i in range(max_alternatives):
            # 这里可以实现更复杂的备选路径算法
            # 暂时返回空列表，实际应用中可以实现
            pass
        
        return alternatives
    
    def recalculate_route(self, route_id: str, force_recalculation: bool = False) -> bool:
        """重新计算路径"""
        with self.lock:
            if route_id not in self.active_routes:
                return False
            
            route = self.active_routes[route_id]
            current_time = datetime.now()
            
            # 检查是否需要重新计算
            time_since_calculation = (current_time - route.created_time).total_seconds()
            
            if (not force_recalculation and 
                time_since_calculation < self.max_recalculation_interval):
                return False
            
            # 检查路径稳定性
            if not self._check_route_stability(route):
                self.stats['stability_violations'] += 1
                logger.info(f"路径稳定性检查失败，重新计算: {route_id}")
            
            # 重新规划路径
            if len(route.path) >= 2:
                new_route = self.plan_route(route.path[0], route.path[-1])
                
                if new_route:
                    # 更新路由
                    new_route.route_id = route_id
                    self.active_routes[route_id] = new_route
                    logger.info(f"路径重新计算成功: {route_id}")
                    return True
            
            return False
    
    def _check_route_stability(self, route: RoutePlan) -> bool:
        """检查路径稳定性"""
        if len(route.path) < 2:
            return True
        
        # 检查路径中的关键路段是否受到重大影响
        for segment in route.segments:
            edge_id = f"{segment.start_node}_{segment.end_node}"
            
            # 检查是否有新的严重事件
            current_time = datetime.now()
            for event in self.traffic_events.values():
                if (event.is_active(current_time) and 
                    event.severity in [TrafficSeverity.HIGH, TrafficSeverity.CRITICAL] and
                    edge_id in event.affected_edges):
                    return False
            
            # 检查交通状况是否发生重大变化
            if edge_id in self.traffic_data:
                traffic_data = self.traffic_data[edge_id]
                if traffic_data.congestion_level > 0.8:  # 严重拥堵
                    return False
        
        return True
    
    def monitor_and_update(self):
        """监控并更新活跃路径"""
        routes_to_update = []
        
        for route_id, route in self.active_routes.items():
            if self.recalculate_route(route_id):
                routes_to_update.append(route_id)
        
        if routes_to_update:
            logger.info(f"更新了 {len(routes_to_update)} 条路径")
    
    def start_monitoring(self, interval: int = 60):
        """启动路径监控线程"""
        def monitor_loop():
            while True:
                try:
                    self.monitor_and_update()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"监控线程错误: {e}")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"启动路径监控，间隔: {interval}秒")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                **self.stats,
                'active_routes': len(self.active_routes),
                'traffic_events': len(self.traffic_events),
                'traffic_data_points': len(self.traffic_data),
                'nodes_count': len(self.nodes)
            }
    
    def export_traffic_data(self) -> str:
        """导出交通数据为JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'traffic_data': {
                edge_id: {
                    'current_speed': data.current_speed,
                    'congestion_level': data.congestion_level,
                    'travel_time': data.travel_time
                }
                for edge_id, data in self.traffic_data.items()
            },
            'traffic_events': {
                event_id: {
                    'type': event.event_type.value,
                    'severity': event.severity.value,
                    'location': event.location,
                    'active': event.is_active()
                }
                for event_id, event in self.traffic_events.items()
            }
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


# 使用示例和测试函数
def create_sample_network() -> DynamicPathPlanner:
    """创建示例路网"""
    planner = DynamicPathPlanner()
    
    # 添加节点
    nodes = [
        ("A", (39.9042, 116.4074)),  # 北京天安门
        ("B", (39.9142, 116.4174)),
        ("C", (39.9242, 116.4274)),
        ("D", (39.9342, 116.4374)),
        ("E", (39.9442, 116.4474)),
        ("F", (39.8942, 116.3974)),
    ]
    
    for node_id, coords in nodes:
        planner.add_node(node_id, coords)
    
    # 添加边
    edges = [
        ("A", "B", 2.0),
        ("B", "C", 1.5),
        ("C", "D", 2.5),
        ("A", "F", 1.8),
        ("F", "E", 2.2),
        ("E", "D", 1.0),
        ("B", "E", 3.0),
    ]
    
    for start, end, distance in edges:
        planner.add_edge(start, end, distance)
    
    return planner


def demo_dynamic_planning():
    """动态路径规划演示"""
    print("=== 动态路径规划演示 ===")
    
    # 创建规划器
    planner = create_sample_network()
    
    # 添加实时交通数据
    traffic_data = [
        TrafficData(edge_id="A_B", current_speed=30.0, free_flow_speed=50.0),
        TrafficData(edge_id="B_C", current_speed=45.0, free_flow_speed=50.0),
        TrafficData(edge_id="C_D", current_speed=25.0, free_flow_speed=50.0),
    ]
    
    for data in traffic_data:
        planner.update_traffic_data(data)
    
    # 添加交通事件
    accident = TrafficEvent(
        event_type=TrafficEventType.ACCIDENT,
        severity=TrafficSeverity.HIGH,
        location=(39.9242, 116.4274),
        affected_edges={"C_D"},
        description="交通事故导致道路拥堵"
    )
    planner.add_traffic_event(accident)
    
    # 规划路径
    print("\n1. 初始路径规划 A -> D")
    route = planner.plan_route("A", "D")
    
    if route:
        print(f"路径: {' -> '.join(route.path)}")
        print(f"总距离: {route.total_distance:.2f} km")
        print(f"预计时间: {route.total_time:.2f} 分钟")
        print(f"置信度: {route.confidence_score:.2f}")
        
        # 添加到活跃路径
        planner.active_routes[route.route_id] = route
    else:
        print("路径规划失败")
    
    # 模拟交通状况变化
    print("\n2. 模拟交通状况变化")
    new_traffic = TrafficData(edge_id="A_B", current_speed=15.0, free_flow_speed=50.0)
    planner.update_traffic_data(new_traffic)
    
    # 重新计算路径
    print("重新计算路径...")
    if planner.recalculate_route(route.route_id, force_recalculation=True):
        updated_route = planner.active_routes[route.route_id]
        print(f"新路径: {' -> '.join(updated_route.path)}")
        print(f"新距离: {updated_route.total_distance:.2f} km")
        print(f"新时间: {updated_route.total_time:.2f} 分钟")
    
    # 显示统计信息
    print("\n3. 系统统计信息")
    stats = planner.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 导出交通数据
    print("\n4. 导出交通数据")
    traffic_json = planner.export_traffic_data()
    print("交通数据已导出")
    
    return planner


if __name__ == "__main__":
    demo_dynamic_planning()