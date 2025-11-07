#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标路径规划系统
支持时间、距离、安全性等多目标优化
实现帕累托最优解计算、权重调整和多种多目标优化算法
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import networkx as nx
from scipy.spatial.distance import euclidean
import heapq
import random
from itertools import combinations
import json
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Path:
    """路径数据类"""
    nodes: List[int]
    edges: List[Tuple[int, int]] = field(default_factory=list)
    total_distance: float = 0.0
    total_time: float = 0.0
    safety_score: float = 0.0
    cost: float = 0.0
    objectives: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.objectives:
            self.objectives = {
                'distance': self.total_distance,
                'time': self.total_time,
                'safety': self.safety_score,
                'cost': self.cost
            }


@dataclass
class OptimizationResult:
    """优化结果类"""
    pareto_solutions: List[Path]
    convergence_history: List[List[Path]] = field(default_factory=list)
    execution_time: float = 0.0
    iterations: int = 0
    algorithm_name: str = ""


class Constraint:
    """约束条件基类"""
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def is_satisfied(self, path: Path) -> bool:
        """检查约束是否满足"""
        pass
    
    @abstractmethod
    def penalty(self, path: Path) -> float:
        """计算违反约束的惩罚值"""
        pass


class DistanceConstraint(Constraint):
    """距离约束"""
    def __init__(self, max_distance: float):
        super().__init__("DistanceConstraint")
        self.max_distance = max_distance
    
    def is_satisfied(self, path: Path) -> bool:
        return path.total_distance <= self.max_distance
    
    def penalty(self, path: Path) -> float:
        return max(0, path.total_distance - self.max_distance)


class TimeConstraint(Constraint):
    """时间约束"""
    def __init__(self, max_time: float):
        super().__init__("TimeConstraint")
        self.max_time = max_time
    
    def is_satisfied(self, path: Path) -> bool:
        return path.total_time <= self.max_time
    
    def penalty(self, path: Path) -> float:
        return max(0, path.total_time - self.max_time)


class SafetyConstraint(Constraint):
    """安全性约束"""
    def __init__(self, min_safety: float):
        super().__init__("SafetyConstraint")
        self.min_safety = min_safety
    
    def is_satisfied(self, path: Path) -> bool:
        return path.safety_score >= self.min_safety
    
    def penalty(self, path: Path) -> float:
        return max(0, self.min_safety - path.safety_score)


class ParetoOptimizer:
    """帕累托最优解计算器"""
    
    @staticmethod
    def is_dominated(solution1: Path, solution2: Path, objectives: List[str]) -> bool:
        """判断solution1是否被solution2支配"""
        better_in_any = False
        for obj in objectives:
            val1 = solution1.objectives.get(obj, float('inf'))
            val2 = solution2.objectives.get(obj, float('inf'))
            
            # 对于距离、时间、成本等，越小越好；对于安全性，越大越好
            if obj in ['safety']:
                if val2 > val1:
                    return False
                elif val2 < val1:
                    better_in_any = True
            else:
                if val2 < val1:
                    return False
                elif val2 > val1:
                    better_in_any = True
        
        return better_in_any
    
    @staticmethod
    def find_pareto_frontier(solutions: List[Path], objectives: List[str]) -> List[Path]:
        """找到帕累托前沿"""
        if not solutions:
            return []
        
        pareto_solutions = []
        for i, solution in enumerate(solutions):
            is_dominated = False
            for j, other_solution in enumerate(solutions):
                if i != j and ParetoOptimizer.is_dominated(solution, other_solution, objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(solution)
        
        return pareto_solutions
    
    @staticmethod
    def sort_by_preference(solutions: List[Path], weights: Dict[str, float]) -> List[Path]:
        """根据偏好权重排序解集"""
        def score_function(path: Path) -> float:
            score = 0.0
            for obj, weight in weights.items():
                if obj in path.objectives:
                    if obj == 'safety':
                        score += weight * path.objectives[obj]  # 安全性越大越好
                    else:
                        score += weight / (path.objectives[obj] + 1e-6)  # 其他目标越小越好
            return score
        
        return sorted(solutions, key=score_function, reverse=True)


class MultiObjectiveOptimizer(ABC):
    """多目标优化算法基类"""
    
    def __init__(self, population_size: int = 100, max_generations: int = 100):
        self.population_size = population_size
        self.max_generations = max_generations
        self.convergence_history = []
    
    @abstractmethod
    def optimize(self, graph: nx.Graph, start: int, goal: int, 
                 objectives: List[str], constraints: List[Constraint]) -> OptimizationResult:
        """执行优化"""
        pass


class NSGA2Optimizer(MultiObjectiveOptimizer):
    """NSGA-II多目标优化算法"""
    
    def __init__(self, population_size: int = 100, max_generations: int = 100, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        super().__init__(population_size, max_generations)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def optimize(self, graph: nx.Graph, start: int, goal: int, 
                 objectives: List[str], constraints: List[Constraint]) -> OptimizationResult:
        """执行NSGA-II优化"""
        import time
        start_time = time.time()
        
        # 初始化种群
        population = self._initialize_population(graph, start, goal, constraints)
        self.convergence_history = [population.copy()]
        
        for generation in range(self.max_generations):
            # 选择、交叉、变异
            offspring = self._genetic_operations(population, graph, start, goal, constraints)
            
            # 合并父代和子代
            combined_population = population + offspring
            
            # 非支配排序和拥挤距离计算
            pareto_fronts = self._non_dominated_sorting(combined_population, objectives)
            
            # 选择下一代种群
            population = self._environmental_selection(combined_population, pareto_fronts, objectives)
            
            self.convergence_history.append(population.copy())
            
            # 早停条件
            if len(pareto_fronts[0]) == 1:
                break
        
        # 提取帕累托前沿
        pareto_fronts = self._non_dominated_sorting(population, objectives)
        pareto_solutions = pareto_fronts[0] if pareto_fronts else []
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            pareto_solutions=pareto_solutions,
            convergence_history=self.convergence_history,
            execution_time=execution_time,
            iterations=generation + 1,
            algorithm_name="NSGA-II"
        )
    
    def _initialize_population(self, graph: nx.Graph, start: int, goal: int, 
                              constraints: List[Constraint]) -> List[Path]:
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            # 使用不同的启发式方法生成初始路径
            if random.random() < 0.5:
                path = self._shortest_path_dijkstra(graph, start, goal)
            else:
                path = self._random_path(graph, start, goal)
            
            if path and self._is_feasible(path, constraints):
                population.append(path)
        
        # 确保种群大小
        while len(population) < self.population_size:
            path = self._random_path(graph, start, goal)
            if path and self._is_feasible(path, constraints):
                population.append(path)
        
        return population[:self.population_size]
    
    def _shortest_path_dijkstra(self, graph: nx.Graph, start: int, goal: int) -> Optional[Path]:
        """使用Dijkstra算法找到最短路径"""
        try:
            shortest_path = nx.shortest_path(graph, start, goal, weight='weight')
            return self._create_path_from_nodes(graph, shortest_path)
        except nx.NetworkXNoPath:
            return None
    
    def _random_path(self, graph: nx.Graph, start: int, goal: int) -> Optional[Path]:
        """生成随机路径"""
        try:
            # 随机游走直到到达目标
            current = start
            path_nodes = [current]
            visited = set([current])
            
            max_steps = len(graph.nodes) * 2
            steps = 0
            
            while current != goal and steps < max_steps:
                neighbors = list(graph.neighbors(current))
                if not neighbors:
                    break
                
                # 优先选择未访问的邻居
                unvisited_neighbors = [n for n in neighbors if n not in visited]
                if unvisited_neighbors:
                    current = random.choice(unvisited_neighbors)
                else:
                    current = random.choice(neighbors)
                
                path_nodes.append(current)
                visited.add(current)
                steps += 1
            
            if current == goal:
                return self._create_path_from_nodes(graph, path_nodes)
            else:
                return None
        except:
            return None
    
    def _create_path_from_nodes(self, graph: nx.Graph, nodes: List[int]) -> Path:
        """从节点列表创建路径对象"""
        edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
        
        total_distance = sum(graph.edges[edge].get('distance', 1.0) for edge in edges)
        total_time = sum(graph.edges[edge].get('time', 1.0) for edge in edges)
        safety_score = np.mean([graph.edges[edge].get('safety', 0.5) for edge in edges])
        cost = sum(graph.edges[edge].get('cost', 1.0) for edge in edges)
        
        path = Path(
            nodes=nodes.copy(),
            edges=edges,
            total_distance=total_distance,
            total_time=total_time,
            safety_score=safety_score,
            cost=cost
        )
        
        return path
    
    def _is_feasible(self, path: Path, constraints: List[Constraint]) -> bool:
        """检查路径是否可行"""
        for constraint in constraints:
            if not constraint.is_satisfied(path):
                return False
        return True
    
    def _genetic_operations(self, population: List[Path], graph: nx.Graph, 
                           start: int, goal: int, constraints: List[Constraint]) -> List[Path]:
        """遗传操作：选择、交叉、变异"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # 选择两个父代
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, graph, start, goal)
            else:
                child1, child2 = parent1, parent2
            
            # 变异
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1, graph, constraints)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2, graph, constraints)
            
            # 添加可行子代
            if child1 and self._is_feasible(child1, constraints):
                offspring.append(child1)
            if child2 and self._is_feasible(child2, constraints):
                offspring.append(child2)
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self, population: List[Path], tournament_size: int = 3) -> Path:
        """锦标赛选择"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda x: x.total_distance)
    
    def _crossover(self, parent1: Path, parent2: Path, graph: nx.Graph, 
                   start: int, goal: int) -> Tuple[Optional[Path], Optional[Path]]:
        """交叉操作"""
        # 找到共同节点
        common_nodes = set(parent1.nodes) & set(parent2.nodes)
        if len(common_nodes) < 2:
            return parent1, parent2
        
        crossover_point = random.choice(list(common_nodes))
        if crossover_point == start or crossover_point == goal:
            return parent1, parent2
        
        # 交换路径段
        idx1 = parent1.nodes.index(crossover_point)
        idx2 = parent2.nodes.index(crossover_point)
        
        child1_nodes = parent1.nodes[:idx1+1] + parent2.nodes[idx2+1:]
        child2_nodes = parent2.nodes[:idx2+1] + parent1.nodes[idx1+1:]
        
        # 验证路径连通性
        try:
            child1_path = self._create_path_from_nodes(graph, child1_nodes)
            child2_path = self._create_path_from_nodes(graph, child2_nodes)
            return child1_path, child2_path
        except:
            return parent1, parent2
    
    def _mutate(self, path: Path, graph: nx.Graph, constraints: List[Constraint]) -> Optional[Path]:
        """变异操作"""
        if len(path.nodes) < 3:
            return path
        
        # 随机选择一个内部节点进行变异
        mutation_idx = random.randint(1, len(path.nodes) - 2)
        current_node = path.nodes[mutation_idx]
        
        # 找到邻居节点
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            return path
        
        # 随机选择一个邻居
        new_node = random.choice(neighbors)
        
        # 创建新路径
        new_nodes = path.nodes[:mutation_idx] + [new_node] + path.nodes[mutation_idx+1:]
        
        try:
            new_path = self._create_path_from_nodes(graph, new_nodes)
            return new_path
        except:
            return path
    
    def _non_dominated_sorting(self, population: List[Path], objectives: List[str]) -> List[List[Path]]:
        """非支配排序"""
        fronts = [[]]
        domination_count = {}
        dominated_solutions = {}
        
        # 计算每个解的支配关系
        for i, solution in enumerate(population):
            domination_count[i] = 0
            dominated_solutions[i] = []
            
            for j, other_solution in enumerate(population):
                if i != j:
                    if ParetoOptimizer.is_dominated(solution, other_solution, objectives):
                        dominated_solutions[i].append(j)
                    elif ParetoOptimizer.is_dominated(other_solution, solution, objectives):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                solution.rank = 0
                fronts[0].append(solution)
        
        # 构建后续前沿
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for solution in fronts[front_idx]:
                solution_idx = population.index(solution)
                for dominated_idx in dominated_solutions[solution_idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        population[dominated_idx].rank = front_idx + 1
                        next_front.append(population[dominated_idx])
            
            front_idx += 1
            fronts.append(next_front)
        
        return [front for front in fronts if front]
    
    def _environmental_selection(self, population: List[Path], fronts: List[List[Path]], 
                                objectives: List[str]) -> List[Path]:
        """环境选择"""
        selected = []
        
        # 添加完整的帕累托前沿
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                # 需要从当前前沿中选择部分解
                remaining_slots = self.population_size - len(selected)
                selected.extend(self._crowding_distance_selection(front, remaining_slots, objectives))
                break
        
        return selected
    
    def _crowding_distance_selection(self, front: List[Path], k: int, objectives: List[str]) -> List[Path]:
        """拥挤距离选择"""
        if len(front) <= k:
            return front
        
        # 计算每个目标的边界值
        objective_ranges = {}
        for obj in objectives:
            values = [solution.objectives.get(obj, 0) for solution in front]
            objective_ranges[obj] = (min(values), max(values))
        
        # 计算拥挤距离
        distances = {i: 0.0 for i in range(len(front))}
        
        for obj in objectives:
            sorted_indices = sorted(range(len(front)), 
                                  key=lambda i: front[i].objectives.get(obj, 0))
            
            # 边界点距离设为无穷大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # 计算中间点的拥挤距离
            obj_range = objective_ranges[obj][1] - objective_ranges[obj][0]
            if obj_range == 0:
                continue
            
            for i in range(1, len(sorted_indices) - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                
                distance_diff = (front[next_idx].objectives.get(obj, 0) - 
                               front[prev_idx].objectives.get(obj, 0)) / obj_range
                distances[idx] += distance_diff
        
        # 根据拥挤距离排序并选择
        sorted_by_distance = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
        return [front[i] for i in sorted_by_distance[:k]]


class MOEADOptimizer(MultiObjectiveOptimizer):
    """MOEA/D多目标优化算法"""
    
    def __init__(self, population_size: int = 100, max_generations: int = 100, 
                 neighborhood_size: int = 20):
        super().__init__(population_size, max_generations)
        self.neighborhood_size = neighborhood_size
        self.weight_vectors = []
        self.neighborhoods = []
    
    def optimize(self, graph: nx.Graph, start: int, goal: int, 
                 objectives: List[str], constraints: List[Constraint]) -> OptimizationResult:
        """执行MOEA/D优化"""
        import time
        start_time = time.time()
        
        # 生成权重向量
        self._generate_weight_vectors(len(objectives))
        
        # 初始化种群
        population = self._initialize_population(graph, start, goal, constraints)
        self.convergence_history = [population.copy()]
        
        # 计算邻域
        self._compute_neighborhoods()
        
        for generation in range(self.max_generations):
            for i in range(self.population_size):
                # 从邻域中选择父母
                parents = self._select_parents(i)
                
                # 生成子代
                offspring = self._differential_evolution(parents[0], parents[1], graph, start, goal, constraints)
                
                # 更新邻域
                self._update_neighbors(i, offspring, objectives)
            
            self.convergence_history.append(population.copy())
        
        # 提取帕累托前沿
        pareto_solutions = ParetoOptimizer.find_pareto_frontier(population, objectives)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            pareto_solutions=pareto_solutions,
            convergence_history=self.convergence_history,
            execution_time=execution_time,
            iterations=generation + 1,
            algorithm_name="MOEA/D"
        )
    
    def _generate_weight_vectors(self, num_objectives: int):
        """生成权重向量"""
        # 简化的权重向量生成
        self.weight_vectors = []
        for i in range(self.population_size):
            weights = np.random.dirichlet(np.ones(num_objectives))
            self.weight_vectors.append(weights)
    
    def _initialize_population(self, graph: nx.Graph, start: int, goal: int, 
                              constraints: List[Constraint]) -> List[Path]:
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            path = self._generate_random_path(graph, start, goal)
            if path and self._is_feasible(path, constraints):
                population.append(path)
        
        return population
    
    def _generate_random_path(self, graph: nx.Graph, start: int, goal: int) -> Optional[Path]:
        """生成随机路径"""
        try:
            # 使用简单的随机游走
            current = start
            path_nodes = [current]
            
            max_steps = len(graph.nodes) * 3
            steps = 0
            
            while current != goal and steps < max_steps:
                neighbors = list(graph.neighbors(current))
                if not neighbors:
                    break
                
                current = random.choice(neighbors)
                path_nodes.append(current)
                steps += 1
            
            if current == goal:
                return self._create_path_from_nodes(graph, path_nodes)
            else:
                return None
        except:
            return None
    
    def _create_path_from_nodes(self, graph: nx.Graph, nodes: List[int]) -> Path:
        """从节点列表创建路径对象"""
        edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
        
        total_distance = sum(graph.edges[edge].get('distance', 1.0) for edge in edges)
        total_time = sum(graph.edges[edge].get('time', 1.0) for edge in edges)
        safety_score = np.mean([graph.edges[edge].get('safety', 0.5) for edge in edges])
        cost = sum(graph.edges[edge].get('cost', 1.0) for edge in edges)
        
        return Path(
            nodes=nodes.copy(),
            edges=edges,
            total_distance=total_distance,
            total_time=total_time,
            safety_score=safety_score,
            cost=cost
        )
    
    def _is_feasible(self, path: Path, constraints: List[Constraint]) -> bool:
        """检查路径是否可行"""
        for constraint in constraints:
            if not constraint.is_satisfied(path):
                return False
        return True
    
    def _compute_neighborhoods(self):
        """计算邻域"""
        self.neighborhoods = []
        for i in range(self.population_size):
            distances = []
            for j in range(self.population_size):
                if i != j:
                    dist = np.linalg.norm(self.weight_vectors[i] - self.weight_vectors[j])
                    distances.append((dist, j))
            
            distances.sort()
            neighborhood = [idx for _, idx in distances[:self.neighborhood_size]]
            self.neighborhoods.append(neighborhood)
    
    def _select_parents(self, index: int) -> List[Path]:
        """选择父母"""
        neighborhood = self.neighborhoods[index]
        parents = random.sample(neighborhood, min(2, len(neighborhood)))
        return parents
    
    def _differential_evolution(self, parent1: Path, parent2: Path, graph: nx.Graph, 
                               start: int, goal: int, constraints: List[Constraint]) -> Optional[Path]:
        """差分进化"""
        # 简化的差分进化操作
        try:
            # 随机选择一个交叉点
            if len(parent1.nodes) > 2 and len(parent2.nodes) > 2:
                crossover_point = random.randint(1, min(len(parent1.nodes), len(parent2.nodes)) - 1)
                
                # 合并路径
                child_nodes = parent1.nodes[:crossover_point] + parent2.nodes[crossover_point:]
                
                # 确保路径连通性
                if len(child_nodes) > 1:
                    return self._create_path_from_nodes(graph, child_nodes)
            
            return parent1
        except:
            return parent1
    
    def _update_neighbors(self, index: int, offspring: Optional[Path], objectives: List[str]):
        """更新邻域"""
        if offspring is None:
            return
        
        neighborhood = self.neighborhoods[index]
        
        for neighbor_idx in neighborhood:
            # 简化的更新规则
            if random.random() < 0.1:  # 更新概率
                pass  # 在实际实现中需要比较目标函数值


class MultiObjectivePathPlanner:
    """多目标路径规划器主类"""
    
    def __init__(self, algorithm: str = "NSGA-II", population_size: int = 100, 
                 max_generations: int = 100):
        self.algorithm = algorithm
        self.population_size = population_size
        self.max_generations = max_generations
        self.optimizer = self._create_optimizer()
        self.constraints = []
        self.objectives = ['distance', 'time', 'safety', 'cost']
        self.weights = {'distance': 0.3, 'time': 0.3, 'safety': 0.2, 'cost': 0.2}
    
    def _create_optimizer(self) -> MultiObjectiveOptimizer:
        """创建优化器"""
        if self.algorithm.upper() == "NSGA-II":
            return NSGA2Optimizer(self.population_size, self.max_generations)
        elif self.algorithm.upper() in ["MOEAD", "MOEA/D"]:
            return MOEADOptimizer(self.population_size, self.max_generations)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
    
    def add_constraint(self, constraint: Constraint):
        """添加约束条件"""
        self.constraints.append(constraint)
    
    def set_objectives(self, objectives: List[str]):
        """设置优化目标"""
        self.objectives = objectives
    
    def set_weights(self, weights: Dict[str, float]):
        """设置权重"""
        self.weights.update(weights)
    
    def plan_path(self, graph: nx.Graph, start: int, goal: int) -> OptimizationResult:
        """规划路径"""
        if start not in graph.nodes or goal not in graph.nodes:
            raise ValueError("起点或终点不在图中")
        
        if not nx.has_path(graph, start, goal):
            raise ValueError("起点和终点之间没有连通路径")
        
        # 执行优化
        result = self.optimizer.optimize(graph, start, goal, self.objectives, self.constraints)
        
        # 根据权重排序解集
        if result.pareto_solutions:
            result.pareto_solutions = ParetoOptimizer.sort_by_preference(
                result.pareto_solutions, self.weights
            )
        
        return result
    
    def get_best_solution(self, result: OptimizationResult, preference: str = "balanced") -> Optional[Path]:
        """获取最佳解"""
        if not result.pareto_solutions:
            return None
        
        if preference == "balanced":
            # 平衡所有目标
            return result.pareto_solutions[0]
        elif preference == "shortest":
            # 最短距离
            return min(result.pareto_solutions, key=lambda x: x.total_distance)
        elif preference == "fastest":
            # 最快时间
            return min(result.pareto_solutions, key=lambda x: x.total_time)
        elif preference == "safest":
            # 最安全
            return max(result.pareto_solutions, key=lambda x: x.safety_score)
        elif preference == "cheapest":
            # 最便宜
            return min(result.pareto_solutions, key=lambda x: x.cost)
        else:
            return result.pareto_solutions[0]
    
    def analyze_solutions(self, result: OptimizationResult) -> Dict[str, Any]:
        """分析解集"""
        if not result.pareto_solutions:
            return {}
        
        solutions = result.pareto_solutions
        
        analysis = {
            'num_solutions': len(solutions),
            'objective_ranges': {},
            'solution_diversity': {},
            'convergence_info': {
                'algorithm': result.algorithm_name,
                'iterations': result.iterations,
                'execution_time': result.execution_time
            }
        }
        
        # 分析各目标范围
        for obj in self.objectives:
            values = [s.objectives.get(obj, 0) for s in solutions]
            analysis['objective_ranges'][obj] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # 计算解集多样性
        if len(solutions) > 1:
            distances = []
            for i in range(len(solutions)):
                for j in range(i + 1, len(solutions)):
                    # 计算解之间的欧氏距离
                    vec1 = [solutions[i].objectives.get(obj, 0) for obj in self.objectives]
                    vec2 = [solutions[j].objectives.get(obj, 0) for obj in self.objectives]
                    dist = euclidean(vec1, vec2)
                    distances.append(dist)
            
            analysis['solution_diversity'] = {
                'mean_distance': np.mean(distances),
                'min_distance': min(distances),
                'max_distance': max(distances)
            }
        
        return analysis


class PathVisualizer:
    """路径可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
    
    def plot_pareto_frontier(self, result: OptimizationResult, objectives: List[str] = None, 
                           save_path: str = None):
        """绘制帕累托前沿"""
        if not result.pareto_solutions:
            print("没有找到帕累托解")
            return
        
        solutions = result.pareto_solutions
        if objectives is None:
            objectives = ['distance', 'time', 'safety', 'cost']
        
        # 创建子图
        n_objectives = len(objectives)
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()
        
        # 绘制目标对之间的散点图
        for i in range(min(6, len(objectives))):
            if i >= n_objectives:
                axes[i].axis('off')
                continue
            
            obj1 = objectives[i]
            for j in range(i + 1, min(i + 2, n_objectives)):
                obj2 = objectives[j]
                
                x_values = [s.objectives.get(obj1, 0) for s in solutions]
                y_values = [s.objectives.get(obj2, 0) for s in solutions]
                
                axes[i].scatter(x_values, y_values, alpha=0.7, s=50)
                axes[i].set_xlabel(obj1)
                axes[i].set_ylabel(obj2)
                axes[i].set_title(f'{obj1} vs {obj2}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_convergence(self, result: OptimizationResult, objective: str = 'distance'):
        """绘制收敛曲线"""
        if not result.convergence_history:
            print("没有收敛历史数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制目标值随代数变化
        generations = range(len(result.convergence_history))
        
        best_values = []
        mean_values = []
        
        for generation in result.convergence_history:
            if generation:
                values = [getattr(s, f'total_{objective}', s.objectives.get(objective, 0)) 
                         for s in generation]
                best_values.append(min(values))
                mean_values.append(np.mean(values))
        
        ax1.plot(generations, best_values, 'b-', label=f'最佳{objective}')
        ax1.plot(generations, mean_values, 'r--', label=f'平均{objective}')
        ax1.set_xlabel('代数')
        ax1.set_ylabel(objective)
        ax1.set_title(f'{objective}收敛曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制解集大小变化
        solution_counts = [len(gen) for gen in result.convergence_history]
        ax2.plot(generations, solution_counts, 'g-', marker='o')
        ax2.set_xlabel('代数')
        ax2.set_ylabel('解集大小')
        ax2.set_title('种群大小变化')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_path_on_graph(self, graph: nx.Graph, paths: List[Path], 
                          save_path: str = None, title: str = "路径规划结果"):
        """在图上绘制路径"""
        if not paths:
            print("没有路径可绘制")
            return
        
        plt.figure(figsize=self.figsize)
        
        # 绘制图的基本结构
        pos = nx.spring_layout(graph, seed=42)
        
        # 绘制所有节点和边
        nx.draw_networkx_nodes(graph, pos, node_color='lightgray', 
                              node_size=300, alpha=0.7)
        nx.draw_networkx_edges(graph, pos, edge_color='gray', 
                              alpha=0.5, width=0.5)
        
        # 绘制路径
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, path in enumerate(paths[:5]):  # 最多绘制5条路径
            color = colors[i % len(colors)]
            path_edges = [(path.nodes[j], path.nodes[j+1]) for j in range(len(path.nodes)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, 
                                  edge_color=color, width=3, alpha=0.8)
        
        # 添加节点标签
        nx.draw_networkx_labels(graph, pos, font_size=8)
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_objective_space_3d(self, result: OptimizationResult, 
                               objectives: List[str] = None, save_path: str = None):
        """绘制三维目标空间"""
        if not result.pareto_solutions:
            print("没有找到帕累托解")
            return
        
        solutions = result.pareto_solutions
        if objectives is None or len(objectives) < 3:
            objectives = ['distance', 'time', 'safety'][:3]
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        x_values = [s.objectives.get(objectives[0], 0) for s in solutions]
        y_values = [s.objectives.get(objectives[1], 0) for s in solutions]
        z_values = [s.objectives.get(objectives[2], 0) for s in solutions]
        
        scatter = ax.scatter(x_values, y_values, z_values, c=z_values, 
                           cmap='viridis', s=50, alpha=0.7)
        
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        ax.set_title('三维目标空间中的帕累托前沿')
        
        plt.colorbar(scatter)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_sample_graph(num_nodes: int = 20, edge_probability: float = 0.3) -> nx.Graph:
    """创建示例图"""
    graph = nx.Graph()
    
    # 添加节点
    for i in range(num_nodes):
        graph.add_node(i)
    
    # 添加边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                # 生成随机属性
                distance = random.uniform(1, 10)
                time = distance * random.uniform(0.5, 2.0)  # 时间与距离相关
                safety = random.uniform(0.1, 1.0)  # 安全性
                cost = distance * random.uniform(0.8, 1.5)  # 成本与距离相关
                
                graph.add_edge(i, j, 
                             distance=distance,
                             time=time,
                             safety=safety,
                             cost=cost,
                             weight=distance)  # 用于Dijkstra算法
    
    # 确保图是连通的
    if not nx.is_connected(graph):
        # 添加边连接各个连通分量
        components = list(nx.connected_components(graph))
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i + 1]))
            
            distance = random.uniform(1, 10)
            time = distance * random.uniform(0.5, 2.0)
            safety = random.uniform(0.1, 1.0)
            cost = distance * random.uniform(0.8, 1.5)
            
            graph.add_edge(node1, node2,
                         distance=distance,
                         time=time,
                         safety=safety,
                         cost=cost,
                         weight=distance)
    
    return graph


if __name__ == "__main__":
    # 创建示例图
    print("创建示例图...")
    graph = create_sample_graph(num_nodes=15, edge_probability=0.4)
    print(f"图包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边")
    
    # 创建多目标路径规划器
    planner = MultiObjectivePathPlanner(algorithm="NSGA-II", population_size=50, max_generations=50)
    
    # 添加约束条件
    max_distance = 30.0
    max_time = 40.0
    min_safety = 0.3
    
    planner.add_constraint(DistanceConstraint(max_distance))
    planner.add_constraint(TimeConstraint(max_time))
    planner.add_constraint(SafetyConstraint(min_safety))
    
    # 设置目标和权重
    planner.set_objectives(['distance', 'time', 'safety', 'cost'])
    planner.set_weights({
        'distance': 0.25,
        'time': 0.25,
        'safety': 0.3,
        'cost': 0.2
    })
    
    # 规划路径
    start_node = 0
    goal_node = graph.number_of_nodes() - 1
    
    print(f"从节点 {start_node} 到节点 {goal_node} 进行多目标路径规划...")
    
    result = planner.plan_path(graph, start_node, goal_node)
    
    print(f"\n优化结果:")
    print(f"算法: {result.algorithm_name}")
    print(f"迭代次数: {result.iterations}")
    print(f"执行时间: {result.execution_time:.2f} 秒")
    print(f"找到帕累托解数量: {len(result.pareto_solutions)}")
    
    if result.pareto_solutions:
        print(f"\n前5个最优解:")
        for i, path in enumerate(result.pareto_solutions[:5]):
            print(f"解 {i+1}:")
            print(f"  路径: {' -> '.join(map(str, path.nodes))}")
            print(f"  距离: {path.total_distance:.2f}")
            print(f"  时间: {path.total_time:.2f}")
            print(f"  安全性: {path.safety_score:.3f}")
            print(f"  成本: {path.cost:.2f}")
            print()
        
        # 分析解集
        analysis = planner.analyze_solutions(result)
        print("解集分析:")
        for obj, stats in analysis['objective_ranges'].items():
            print(f"{obj}: 最小值={stats['min']:.2f}, 最大值={stats['max']:.2f}, "
                  f"平均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}")
        
        # 获取不同偏好的最佳解
        preferences = ['balanced', 'shortest', 'fastest', 'safest', 'cheapest']
        print(f"\n不同偏好的最佳解:")
        for pref in preferences:
            best_path = planner.get_best_solution(result, pref)
            if best_path:
                print(f"{pref}: 距离={best_path.total_distance:.2f}, "
                      f"时间={best_path.total_time:.2f}, "
                      f"安全性={best_path.safety_score:.3f}, "
                      f"成本={best_path.cost:.2f}")
        
        # 可视化结果
        print("\n生成可视化图表...")
        visualizer = PathVisualizer()
        
        # 绘制帕累托前沿
        visualizer.plot_pareto_frontier(result, save_path='pareto_frontier.png')
        
        # 绘制收敛曲线
        visualizer.plot_convergence(result, 'distance')
        
        # 在图上绘制路径
        visualizer.plot_path_on_graph(graph, result.pareto_solutions[:3], 
                                    save_path='optimal_paths.png')
        
        # 绘制三维目标空间
        visualizer.plot_objective_space_3d(result, save_path='objective_space_3d.png')
        
        print("可视化完成！")
    else:
        print("未找到可行解，请调整约束条件或参数。")