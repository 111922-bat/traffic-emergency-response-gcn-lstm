"""
救援车辆调度优化系统
Emergency Vehicle Dispatch Optimization System

支持多车辆管理、任务分配、负载均衡、动态调度等功能
"""

import time
import heapq
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import math
import json
from collections import defaultdict, deque
import threading


class VehicleType(Enum):
    """车辆类型枚举"""
    MEDICAL = "医疗救援"
    FIRE = "消防救援"
    POLICE = "警察"
    RESCUE = "搜救"
    AMBULANCE = "救护车"
    HAZMAT = "危险品处理"


class VehicleStatus(Enum):
    """车辆状态枚举"""
    AVAILABLE = "可用"
    BUSY = "执行任务中"
    MAINTENANCE = "维护中"
    OFFLINE = "离线"
    EN_ROUTE = "前往现场"
    ON_SCENE = "到达现场"
    RETURNING = "返回基地"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "等待分配"
    ASSIGNED = "已分配"
    IN_PROGRESS = "执行中"
    COMPLETED = "已完成"
    CANCELLED = "已取消"
    FAILED = "失败"


@dataclass
class Location:
    """位置信息"""
    x: float
    y: float
    name: Optional[str] = None
    
    def distance_to(self, other: 'Location') -> float:
        """计算到另一个位置的距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "name": self.name}


@dataclass
class EmergencyTask:
    """紧急任务"""
    task_id: str
    location: Location
    task_type: VehicleType
    priority: TaskPriority
    description: str
    created_time: datetime = field(default_factory=datetime.now)
    estimated_duration: int = 30  # 预估持续时间（分钟）
    required_vehicles: int = 1
    special_requirements: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_vehicles: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "location": self.location.to_dict(),
            "task_type": self.task_type.value,
            "priority": self.priority.value,
            "description": self.description,
            "created_time": self.created_time.isoformat(),
            "estimated_duration": self.estimated_duration,
            "required_vehicles": self.required_vehicles,
            "special_requirements": self.special_requirements,
            "status": self.status.value,
            "assigned_vehicles": self.assigned_vehicles,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None
        }


@dataclass
class EmergencyVehicle:
    """救援车辆"""
    vehicle_id: str
    vehicle_type: VehicleType
    current_location: Location
    status: VehicleStatus = VehicleStatus.AVAILABLE
    max_capacity: int = 1
    current_load: int = 0
    fuel_level: float = 100.0
    maintenance_due: bool = False
    assigned_tasks: List[str] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    response_time_history: List[float] = field(default_factory=list)
    efficiency_score: float = 100.0
    
    def is_available(self) -> bool:
        """检查车辆是否可用"""
        return (self.status == VehicleStatus.AVAILABLE and 
                self.fuel_level > 20.0 and 
                not self.maintenance_due)
    
    def can_handle_task(self, task: EmergencyTask) -> bool:
        """检查车辆是否可以处理指定任务"""
        if not self.is_available():
            return False
        
        if self.vehicle_type != task.task_type:
            # 特殊处理：医疗救援车可以处理救护车任务
            if not (self.vehicle_type == VehicleType.MEDICAL and task.task_type == VehicleType.AMBULANCE):
                return False
        
        if self.current_load + task.required_vehicles > self.max_capacity:
            return False
            
        return True
    
    def update_location(self, new_location: Location):
        """更新车辆位置"""
        self.current_location = new_location
        self.last_update = datetime.now()
    
    def assign_task(self, task_id: str):
        """分配任务"""
        self.assigned_tasks.append(task_id)
        self.current_load += 1
        self.status = VehicleStatus.BUSY
    
    def complete_task(self, task_id: str, response_time: float):
        """完成任务"""
        if task_id in self.assigned_tasks:
            self.assigned_tasks.remove(task_id)
            self.current_load = max(0, self.current_load - 1)
            
            # 更新响应时间历史
            self.response_time_history.append(response_time)
            if len(self.response_time_history) > 50:  # 保持最近50次记录
                self.response_time_history.pop(0)
            
            # 更新效率分数
            avg_response_time = sum(self.response_time_history) / len(self.response_time_history)
            self.efficiency_score = max(0, 100 - avg_response_time / 2)
            
            if self.current_load == 0:
                self.status = VehicleStatus.AVAILABLE
    
    def to_dict(self) -> Dict:
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type.value,
            "current_location": self.current_location.to_dict(),
            "status": self.status.value,
            "max_capacity": self.max_capacity,
            "current_load": self.current_load,
            "fuel_level": self.fuel_level,
            "maintenance_due": self.maintenance_due,
            "assigned_tasks": self.assigned_tasks,
            "last_update": self.last_update.isoformat(),
            "efficiency_score": self.efficiency_score
        }


class DispatchOptimizer:
    """调度优化器"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.optimization_strategies = {
            "closest": self._optimize_closest,
            "load_balanced": self._optimize_load_balanced,
            "priority_first": self._optimize_priority_first,
            "efficiency_based": self._optimize_efficiency_based
        }
        self.current_strategy = "efficiency_based"
    
    def optimize_dispatch(self, vehicles: List[EmergencyVehicle], 
                         task: EmergencyTask) -> List[EmergencyVehicle]:
        """优化调度决策"""
        available_vehicles = [v for v in vehicles if v.can_handle_task(task)]
        
        if not available_vehicles:
            return []
        
        optimizer_func = self.optimization_strategies[self.current_strategy]
        return optimizer_func(available_vehicles, task)
    
    def _optimize_closest(self, vehicles: List[EmergencyVehicle], 
                         task: EmergencyTask) -> List[EmergencyVehicle]:
        """选择最近的车辆"""
        def get_distance(vehicle: EmergencyVehicle) -> float:
            return vehicle.current_location.distance_to(task.location)
        
        sorted_vehicles = sorted(vehicles, key=get_distance)
        return sorted_vehicles[:task.required_vehicles]
    
    def _optimize_load_balanced(self, vehicles: List[EmergencyVehicle], 
                               task: EmergencyTask) -> List[EmergencyVehicle]:
        """负载均衡优化"""
        def get_load_score(vehicle: EmergencyVehicle) -> float:
            return vehicle.current_load / vehicle.max_capacity
        
        sorted_vehicles = sorted(vehicles, key=get_load_score)
        return sorted_vehicles[:task.required_vehicles]
    
    def _optimize_priority_first(self, vehicles: List[EmergencyVehicle], 
                                task: EmergencyTask) -> List[EmergencyVehicle]:
        """优先级优先优化"""
        # 高优先级任务优先选择效率高的车辆
        if task.priority.value >= TaskPriority.HIGH.value:
            def get_efficiency(vehicle: EmergencyVehicle) -> float:
                return vehicle.efficiency_score
            
            sorted_vehicles = sorted(vehicles, key=get_efficiency, reverse=True)
        else:
            sorted_vehicles = sorted(vehicles, key=lambda v: v.current_load)
        
        return sorted_vehicles[:task.required_vehicles]
    
    def _optimize_efficiency_based(self, vehicles: List[EmergencyVehicle], 
                                  task: EmergencyTask) -> List[EmergencyVehicle]:
        """基于效率的优化"""
        def get_efficiency_score(vehicle: EmergencyVehicle) -> float:
            # 综合考虑距离、负载、效率分数和燃料水平
            distance = vehicle.current_location.distance_to(task.location)
            load_factor = vehicle.current_load / vehicle.max_capacity
            fuel_factor = vehicle.fuel_level / 100.0
            
            # 距离越近、负载越轻、效率越高、燃料越充足，得分越高
            score = (vehicle.efficiency_score * 0.4 + 
                    (100 - distance) * 0.3 + 
                    (1 - load_factor) * 100 * 0.2 + 
                    fuel_factor * 100 * 0.1)
            return score
        
        sorted_vehicles = sorted(vehicles, key=get_efficiency_score, reverse=True)
        return sorted_vehicles[:task.required_vehicles]
    
    def update_performance(self, dispatch_result: Dict):
        """更新性能数据"""
        self.performance_history.append(dispatch_result)
    
    def get_optimization_suggestion(self) -> str:
        """获取优化建议"""
        if len(self.performance_history) < 10:
            return "需要更多数据来提供优化建议"
        
        recent_performance = list(self.performance_history)[-10:]
        avg_response_time = sum(p.get("response_time", 0) for p in recent_performance) / len(recent_performance)
        completion_rate = sum(1 for p in recent_performance if p.get("completed", False)) / len(recent_performance)
        
        if avg_response_time > 15:  # 平均响应时间超过15分钟
            return "建议切换到最近车辆优先策略以减少响应时间"
        elif completion_rate < 0.8:  # 任务完成率低于80%
            return "建议切换到效率优先策略以提高任务完成率"
        else:
            return "当前调度策略运行良好"


class EmergencyDispatcher:
    """救援车辆调度器主类"""
    
    def __init__(self, name: str = "Emergency Dispatch Center"):
        self.name = name
        self.vehicles: Dict[str, EmergencyVehicle] = {}
        self.tasks: Dict[str, EmergencyTask] = {}
        self.task_queue: List[EmergencyTask] = []
        self.optimizer = DispatchOptimizer()
        self.dispatch_history: List[Dict] = []
        self.is_running = False
        self.dispatch_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # 性能统计
        self.stats = {
            "total_dispatches": 0,
            "successful_dispatches": 0,
            "average_response_time": 0.0,
            "tasks_completed": 0,
            "tasks_failed": 0
        }
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_vehicle(self, vehicle: EmergencyVehicle):
        """添加车辆"""
        with self.lock:
            self.vehicles[vehicle.vehicle_id] = vehicle
            self.logger.info(f"添加车辆: {vehicle.vehicle_id} ({vehicle.vehicle_type.value})")
    
    def remove_vehicle(self, vehicle_id: str):
        """移除车辆"""
        with self.lock:
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles.pop(vehicle_id)
                self.logger.info(f"移除车辆: {vehicle_id}")
    
    def add_task(self, task: EmergencyTask):
        """添加任务"""
        with self.lock:
            self.tasks[task.task_id] = task
            heapq.heappush(self.task_queue, (task.priority.value, task.created_time, task))
            self.logger.info(f"添加任务: {task.task_id} - {task.description}")
    
    def get_available_vehicles(self, task_type: VehicleType = None) -> List[EmergencyVehicle]:
        """获取可用车辆"""
        with self.lock:
            available = [v for v in self.vehicles.values() if v.is_available()]
            if task_type:
                available = [v for v in available if v.vehicle_type == task_type or 
                           (v.vehicle_type == VehicleType.MEDICAL and task_type == VehicleType.AMBULANCE)]
            return available
    
    def dispatch_task(self, task_id: str) -> bool:
        """调度任务"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.status != TaskStatus.PENDING:
                return False
            
            # 优化选择车辆
            selected_vehicles = self.optimizer.optimize_dispatch(
                list(self.vehicles.values()), task
            )
            
            if not selected_vehicles:
                self.logger.warning(f"没有可用车辆处理任务: {task_id}")
                return False
            
            # 分配任务
            start_time = datetime.now()
            for vehicle in selected_vehicles[:task.required_vehicles]:
                vehicle.assign_task(task_id)
                task.assigned_vehicles.append(vehicle.vehicle_id)
            
            task.status = TaskStatus.ASSIGNED
            task.start_time = start_time
            
            # 记录调度历史
            dispatch_record = {
                "task_id": task_id,
                "vehicles_assigned": [v.vehicle_id for v in selected_vehicles],
                "dispatch_time": start_time.isoformat(),
                "task_priority": task.priority.value,
                "task_type": task.task_type.value
            }
            self.dispatch_history.append(dispatch_record)
            
            self.stats["total_dispatches"] += 1
            self.logger.info(f"任务 {task_id} 已分配给车辆: {[v.vehicle_id for v in selected_vehicles]}")
            return True
    
    def complete_task(self, task_id: str) -> bool:
        """完成任务"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.status not in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                return False
            
            completion_time = datetime.now()
            response_time = (completion_time - task.start_time).total_seconds() / 60
            
            # 更新车辆状态
            for vehicle_id in task.assigned_vehicles:
                if vehicle_id in self.vehicles:
                    vehicle = self.vehicles[vehicle_id]
                    vehicle.complete_task(task_id, response_time)
            
            task.status = TaskStatus.COMPLETED
            task.completion_time = completion_time
            
            # 更新统计
            self.stats["tasks_completed"] += 1
            self.stats["successful_dispatches"] += 1
            
            # 计算平均响应时间
            total_response_time = (self.stats["average_response_time"] * 
                                 (self.stats["successful_dispatches"] - 1) + response_time)
            self.stats["average_response_time"] = total_response_time / self.stats["successful_dispatches"]
            
            # 更新优化器性能数据
            self.optimizer.update_performance({
                "response_time": response_time,
                "completed": True,
                "task_priority": task.priority.value
            })
            
            self.logger.info(f"任务 {task_id} 已完成，响应时间: {response_time:.1f}分钟")
            return True
    
    def auto_dispatch_loop(self):
        """自动调度循环"""
        self.logger.info("启动自动调度循环")
        self.is_running = True
        
        while self.is_running:
            try:
                with self.lock:
                    # 处理待分配的高优先级任务
                    high_priority_tasks = [
                        task for _, _, task in self.task_queue 
                        if task.status == TaskStatus.PENDING and 
                        task.priority.value >= TaskPriority.HIGH.value
                    ]
                    
                    for task in high_priority_tasks:
                        self.dispatch_task(task.task_id)
                    
                    # 处理其他待分配任务
                    medium_priority_tasks = [
                        task for _, _, task in self.task_queue 
                        if task.status == TaskStatus.PENDING and 
                        task.priority.value < TaskPriority.HIGH.value
                    ]
                    
                    for task in medium_priority_tasks:
                        if len(self.get_available_vehicles(task.task_type)) >= task.required_vehicles:
                            self.dispatch_task(task.task_id)
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                self.logger.error(f"自动调度循环错误: {e}")
                time.sleep(5)
    
    def start_auto_dispatch(self):
        """启动自动调度"""
        if not self.is_running:
            self.dispatch_thread = threading.Thread(target=self.auto_dispatch_loop)
            self.dispatch_thread.daemon = True
            self.dispatch_thread.start()
            self.logger.info("自动调度已启动")
    
    def stop_auto_dispatch(self):
        """停止自动调度"""
        self.is_running = False
        if self.dispatch_thread:
            self.dispatch_thread.join()
        self.logger.info("自动调度已停止")
    
    def get_dashboard_data(self) -> Dict:
        """获取调度面板数据"""
        with self.lock:
            available_vehicles = len(self.get_available_vehicles())
            busy_vehicles = len([v for v in self.vehicles.values() if v.status == VehicleStatus.BUSY])
            pending_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
            
            # 按类型统计车辆
            vehicles_by_type = defaultdict(int)
            for vehicle in self.vehicles.values():
                vehicles_by_type[vehicle.vehicle_type.value] += 1
            
            # 按类型统计任务
            tasks_by_type = defaultdict(int)
            for task in self.tasks.values():
                tasks_by_type[task.task_type.value] += 1
            
            return {
                "center_name": self.name,
                "total_vehicles": len(self.vehicles),
                "available_vehicles": available_vehicles,
                "busy_vehicles": busy_vehicles,
                "pending_tasks": pending_tasks,
                "vehicles_by_type": dict(vehicles_by_type),
                "tasks_by_type": dict(tasks_by_type),
                "statistics": self.stats.copy(),
                "optimization_suggestion": self.optimizer.get_optimization_suggestion(),
                "recent_dispatches": self.dispatch_history[-10:] if self.dispatch_history else []
            }
    
    def optimize_strategy(self, strategy: str):
        """切换优化策略"""
        if strategy in self.optimizer.optimization_strategies:
            self.optimizer.current_strategy = strategy
            self.logger.info(f"切换优化策略到: {strategy}")
        else:
            self.logger.warning(f"未知优化策略: {strategy}")
    
    def export_dispatch_data(self, filename: str):
        """导出调度数据"""
        data = {
            "export_time": datetime.now().isoformat(),
            "dashboard_data": self.get_dashboard_data(),
            "vehicles": {vid: v.to_dict() for vid, v in self.vehicles.items()},
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "dispatch_history": self.dispatch_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"调度数据已导出到: {filename}")
    
    def load_dispatch_data(self, filename: str):
        """导入调度数据"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建车辆和任务数据
            self.vehicles.clear()
            self.tasks.clear()
            
            for vid, v_data in data.get("vehicles", {}).items():
                location = Location(**v_data["current_location"])
                vehicle = EmergencyVehicle(
                    vehicle_id=vid,
                    vehicle_type=VehicleType(v_data["vehicle_type"]),
                    current_location=location
                )
                vehicle.status = VehicleStatus(v_data["status"])
                vehicle.max_capacity = v_data["max_capacity"]
                vehicle.current_load = v_data["current_load"]
                vehicle.fuel_level = v_data["fuel_level"]
                vehicle.maintenance_due = v_data["maintenance_due"]
                vehicle.assigned_tasks = v_data["assigned_tasks"]
                vehicle.efficiency_score = v_data["efficiency_score"]
                self.vehicles[vid] = vehicle
            
            for tid, t_data in data.get("tasks", {}).items():
                location = Location(**t_data["location"])
                task = EmergencyTask(
                    task_id=tid,
                    location=location,
                    task_type=VehicleType(t_data["task_type"]),
                    priority=TaskPriority(t_data["priority"]),
                    description=t_data["description"],
                    estimated_duration=t_data["estimated_duration"],
                    required_vehicles=t_data["required_vehicles"],
                    special_requirements=t_data["special_requirements"]
                )
                task.status = TaskStatus(t_data["status"])
                task.assigned_vehicles = t_data["assigned_vehicles"]
                self.tasks[tid] = task
            
            self.dispatch_history = data.get("dispatch_history", [])
            
            self.logger.info(f"调度数据已从 {filename} 导入")
            
        except Exception as e:
            self.logger.error(f"导入调度数据失败: {e}")
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        with self.lock:
            total_tasks = len(self.tasks)
            completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
            failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
            
            # 计算任务类型分布
            task_type_distribution = defaultdict(int)
            for task in self.tasks.values():
                task_type_distribution[task.task_type.value] += 1
            
            # 计算车辆利用率
            vehicle_utilization = {}
            for vehicle in self.vehicles.values():
                total_assignments = len(vehicle.assigned_tasks) + len(vehicle.response_time_history)
                vehicle_utilization[vehicle.vehicle_id] = {
                    "total_assignments": total_assignments,
                    "efficiency_score": vehicle.efficiency_score,
                    "average_response_time": sum(vehicle.response_time_history) / len(vehicle.response_time_history) 
                                            if vehicle.response_time_history else 0
                }
            
            return {
                "report_time": datetime.now().isoformat(),
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
                "average_response_time": self.stats["average_response_time"],
                "task_type_distribution": dict(task_type_distribution),
                "vehicle_utilization": vehicle_utilization,
                "optimization_performance": {
                    "strategy": self.optimizer.current_strategy,
                    "suggestion": self.optimizer.get_optimization_suggestion()
                }
            }


# 便利函数
def create_sample_dispatcher() -> EmergencyDispatcher:
    """创建示例调度器"""
    dispatcher = EmergencyDispatcher("示例救援调度中心")
    
    # 添加示例车辆
    vehicles = [
        EmergencyVehicle("MED-001", VehicleType.MEDICAL, Location(10, 20, "医疗站1")),
        EmergencyVehicle("MED-002", VehicleType.MEDICAL, Location(30, 40, "医疗站2")),
        EmergencyVehicle("FIRE-001", VehicleType.FIRE, Location(50, 60, "消防站1")),
        EmergencyVehicle("FIRE-002", VehicleType.FIRE, Location(70, 80, "消防站2")),
        EmergencyVehicle("POL-001", VehicleType.POLICE, Location(90, 100, "警察局1")),
        EmergencyVehicle("POL-002", VehicleType.POLICE, Location(110, 120, "警察局2")),
        EmergencyVehicle("RES-001", VehicleType.RESCUE, Location(130, 140, "搜救站1")),
        EmergencyVehicle("AMB-001", VehicleType.AMBULANCE, Location(150, 160, "救护站1")),
        EmergencyVehicle("HAZ-001", VehicleType.HAZMAT, Location(170, 180, "危险品处理站1")),
    ]
    
    for vehicle in vehicles:
        dispatcher.add_vehicle(vehicle)
    
    return dispatcher


if __name__ == "__main__":
    # 创建示例调度器
    dispatcher = create_sample_dispatcher()
    
    # 添加示例任务
    tasks = [
        EmergencyTask("T001", Location(25, 35), VehicleType.MEDICAL, TaskPriority.HIGH, 
                     "心脏病患者需要紧急救治"),
        EmergencyTask("T002", Location(55, 65), VehicleType.FIRE, TaskPriority.EMERGENCY, 
                     "高层建筑火灾"),
        EmergencyTask("T003", Location(95, 105), VehicleType.POLICE, TaskPriority.MEDIUM, 
                     "交通事故处理"),
        EmergencyTask("T004", Location(135, 145), VehicleType.RESCUE, TaskPriority.HIGH, 
                     "山难搜救任务"),
    ]
    
    for task in tasks:
        dispatcher.add_task(task)
    
    # 启动自动调度
    dispatcher.start_auto_dispatch()
    
    try:
        # 运行示例
        time.sleep(2)
        
        # 手动调度任务
        dispatcher.dispatch_task("T001")
        time.sleep(1)
        dispatcher.dispatch_task("T002")
        
        # 完成任务
        time.sleep(2)
        dispatcher.complete_task("T001")
        dispatcher.complete_task("T002")
        
        # 获取性能报告
        report = dispatcher.get_performance_report()
        print("性能报告:")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        
        # 获取面板数据
        dashboard = dispatcher.get_dashboard_data()
        print("\n调度面板数据:")
        print(json.dumps(dashboard, ensure_ascii=False, indent=2))
        
    finally:
        dispatcher.stop_auto_dispatch()