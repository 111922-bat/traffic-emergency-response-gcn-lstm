#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标路径规划系统示例
演示各种应用场景和使用方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_objective_planner import (
    MultiObjectivePathPlanner, NSGA2Optimizer, MOEADOptimizer,
    DistanceConstraint, TimeConstraint, SafetyConstraint,
    Path, OptimizationResult, create_sample_graph,
    ParetoOptimizer, PathVisualizer
)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json


def example_city_navigation():
    """示例1: 城市导航系统"""
    print("=== 示例1: 城市导航系统 ===")
    print("场景: 在城市中从家到公司，需要考虑距离、时间、交通安全和成本")
    
    # 创建城市道路网络
    graph = nx.Graph()
    
    # 城市区域节点 (简化版城市地图)
    locations = {
        0: "家", 1: "学校", 2: "公园", 3: "购物中心", 4: "医院",
        5: "加油站", 6: "餐厅", 7: "银行", 8: "公司", 9: "地铁站",
        10: "图书馆", 11: "电影院", 12: "超市", 13: "咖啡厅", 14: "公司附近"
    }
    
    # 添加道路连接 (distance, time, safety, cost)
    roads = [
        (0, 1, 2.5, 4, 0.8, 3.0),  # 家-学校
        (0, 5, 1.8, 3, 0.9, 2.5),  # 家-加油站
        (1, 2, 3.2, 5, 0.7, 4.0),  # 学校-公园
        (1, 10, 2.8, 4, 0.8, 3.5), # 学校-图书馆
        (2, 3, 4.5, 7, 0.6, 5.5),  # 公园-购物中心
        (3, 12, 1.5, 2, 0.9, 2.0), # 购物中心-超市
        (3, 6, 2.0, 3, 0.8, 2.5),  # 购物中心-餐厅
        (5, 7, 3.0, 5, 0.7, 4.0),  # 加油站-银行
        (7, 8, 5.0, 8, 0.5, 6.0),  # 银行-公司
        (6, 8, 4.8, 7, 0.6, 5.5),  # 餐厅-公司
        (10, 11, 3.5, 6, 0.7, 4.5), # 图书馆-电影院
        (11, 8, 6.0, 9, 0.4, 7.0), # 电影院-公司
        (12, 13, 2.2, 4, 0.8, 3.0), # 超市-咖啡厅
        (13, 14, 1.0, 2, 0.9, 1.5), # 咖啡厅-公司附近
        (14, 8, 0.8, 1, 0.95, 1.0), # 公司附近-公司
        (9, 8, 4.0, 6, 0.6, 5.0),  # 地铁站-公司
        (9, 14, 3.5, 5, 0.7, 4.0), # 地铁站-公司附近
    ]
    
    for start, end, distance, time, safety, cost in roads:
        graph.add_edge(start, end, distance=distance, time=time, safety=safety, cost=cost, weight=distance)
    
    print(f"城市地图: {graph.number_of_nodes()} 个地点, {graph.number_of_edges()} 条道路")
    
    # 创建导航规划器
    planner = MultiObjectivePathPlanner(algorithm="NSGA-II", population_size=60, max_generations=50)
    
    # 设置约束 (假设用户偏好)
    planner.add_constraint(DistanceConstraint(15))  # 不超过15公里
    planner.add_constraint(TimeConstraint(25))     # 不超过25分钟
    planner.add_constraint(SafetyConstraint(0.6))  # 安全性不低于0.6
    
    # 设置权重 (用户偏好: 时间优先)
    planner.set_weights({
        'distance': 0.2,  # 距离权重较低
        'time': 0.5,      # 时间权重最高
        'safety': 0.2,    # 安全性权重中等
        'cost': 0.1       # 成本权重较低
    })
    
    # 规划从家到公司的路径
    home = 0
    company = 8
    
    print(f"\n从 {locations[home]} 到 {locations[company]} 规划路径...")
    result = planner.plan_path(graph, home, company)
    
    if result.pareto_solutions:
        print(f"\n找到 {len(result.pareto_solutions)} 个可行路径方案:")
        
        # 显示所有方案
        for i, path in enumerate(result.pareto_solutions):
            route = " -> ".join([locations[node] for node in path.nodes])
            print(f"\n方案 {i+1}: {route}")
            print(f"  距离: {path.total_distance:.1f} 公里")
            print(f"  时间: {path.total_time:.1f} 分钟")
            print(f"  安全性: {path.safety_score:.2f} (1.0=最安全)")
            print(f"  成本: {path.cost:.1f} 元")
        
        # 获取推荐方案
        recommended = planner.get_best_solution(result, "balanced")
        print(f"\n推荐方案: {' -> '.join([locations[node] for node in recommended.nodes])}")
        print(f"推荐理由: 基于您的时间优先偏好，这是综合最优的方案")
        
        # 分析不同偏好下的方案
        print(f"\n不同偏好下的最佳方案:")
        preferences = {
            "最快": planner.get_best_solution(result, "fastest"),
            "最短": planner.get_best_solution(result, "shortest"), 
            "最安全": planner.get_best_solution(result, "safest"),
            "最便宜": planner.get_best_solution(result, "cheapest")
        }
        
        for pref_name, pref_path in preferences.items():
            if pref_path:
                route = " -> ".join([locations[node] for node in pref_path.nodes])
                print(f"{pref_name}: {route}")
                print(f"  (距离:{pref_path.total_distance:.1f}km, "
                      f"时间:{pref_path.total_time:.1f}min, "
                      f"安全:{pref_path.safety_score:.2f}, "
                      f"成本:{pref_path.cost:.1f}元)")
        
        # 生成可视化
        print(f"\n生成导航可视化...")
        visualizer = PathVisualizer()
        
        # 在地图上显示路径
        visualizer.plot_path_on_graph(
            graph, result.pareto_solutions[:3], 
            save_path='city_navigation_paths.png',
            title="城市导航路径推荐"
        )
        
        # 显示帕累托前沿
        visualizer.plot_pareto_frontier(
            result, save_path='city_navigation_pareto.png'
        )
        
    else:
        print("未找到满足约束的路径，请调整约束条件")


def example_emergency_routing():
    """示例2: 应急救援路径规划"""
    print("\n=== 示例2: 应急救援路径规划 ===")
    print("场景: 紧急情况下从医院到事故地点，需要考虑最短时间和最高安全性")
    
    # 创建应急道路网络
    graph = nx.Graph()
    
    # 应急节点
    emergency_nodes = {
        0: "医院", 1: "主干道1", 2: "主干道2", 3: "事故地点A", 
        4: "事故地点B", 5: "主干道3", 6: "主干道4", 7: "事故地点C",
        8: "消防站", 9: "警察局", 10: "应急指挥中心"
    }
    
    # 应急道路 (更重视时间和安全性)
    emergency_roads = [
        (0, 1, 3.0, 4, 0.9, 4.0),   # 医院-主干道1
        (1, 2, 2.5, 3, 0.8, 3.5),   # 主干道1-主干道2
        (2, 3, 1.5, 2, 0.9, 2.0),   # 主干道2-事故地点A
        (2, 4, 2.0, 3, 0.7, 2.5),   # 主干道2-事故地点B
        (1, 5, 4.0, 5, 0.6, 5.0),   # 主干道1-主干道3
        (5, 6, 3.5, 4, 0.7, 4.0),   # 主干道3-主干道4
        (6, 7, 2.0, 3, 0.8, 2.5),   # 主干道4-事故地点C
        (8, 2, 5.0, 6, 0.5, 6.0),   # 消防站-主干道2
        (9, 5, 4.5, 5, 0.6, 5.5),   # 警察局-主干道3
        (10, 6, 3.0, 4, 0.8, 4.0),  # 应急指挥中心-主干道4
    ]
    
    for start, end, distance, time, safety, cost in emergency_roads:
        graph.add_edge(start, end, distance=distance, time=time, safety=safety, cost=cost, weight=time)
    
    print(f"应急网络: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条道路")
    
    # 创建应急规划器 (更重视时间和安全性)
    planner = MultiObjectivePathPlanner(algorithm="NSGA-II", population_size=40, max_generations=40)
    
    # 应急约束 (更严格的时间要求)
    planner.add_constraint(TimeConstraint(15))     # 必须在15分钟内到达
    planner.add_constraint(SafetyConstraint(0.7))  # 安全性要求更高
    planner.add_constraint(DistanceConstraint(20)) # 距离限制
    
    # 应急权重 (时间和安全性优先)
    planner.set_weights({
        'distance': 0.1,  # 距离权重很低
        'time': 0.6,      # 时间权重最高
        'safety': 0.3,    # 安全性权重很高
        'cost': 0.0       # 成本不考虑
    })
    
    # 规划从医院到不同事故地点的路径
    hospital = 0
    accident_sites = [3, 4, 7]
    
    for site in accident_sites:
        print(f"\n从医院到 {emergency_nodes[site]} 规划应急路径...")
        
        result = planner.plan_path(graph, hospital, site)
        
        if result.pareto_solutions:
            best_path = planner.get_best_solution(result, "fastest")
            
            route = " -> ".join([emergency_nodes[node] for node in best_path.nodes])
            print(f"最佳应急路径: {route}")
            print(f"预计时间: {best_path.total_time:.1f} 分钟")
            print(f"安全性评分: {best_path.safety_score:.2f}")
            print(f"路径距离: {best_path.total_distance:.1f} 公里")
            
            # 如果有多个方案，显示权衡选择
            if len(result.pareto_solutions) > 1:
                print(f"\n备选方案:")
                for i, path in enumerate(result.pareto_solutions[1:3]):
                    route = " -> ".join([emergency_nodes[node] for node in path.nodes])
                    print(f"方案{i+2}: {route}")
                    print(f"  时间:{path.total_time:.1f}min, 安全:{path.safety_score:.2f}")
        else:
            print(f"到 {emergency_nodes[site]} 没有找到可行的应急路径")


def example_logistics_optimization():
    """示例3: 物流配送优化"""
    print("\n=== 示例3: 物流配送优化 ===")
    print("场景: 物流公司规划配送路线，需要平衡距离、时间、成本和客户满意度")
    
    # 创建物流网络
    graph = nx.Graph()
    
    # 物流节点
    logistics_nodes = {
        0: "仓库", 1: "客户A", 2: "客户B", 3: "客户C", 
        4: "客户D", 5: "客户E", 6: "中转站1", 7: "中转站2", 
        8: "加油站", 9: "检查站"
    }
    
    # 物流路线
    logistics_routes = [
        (0, 6, 5.0, 8, 0.8, 6.0),   # 仓库-中转站1
        (0, 7, 6.0, 9, 0.7, 7.0),   # 仓库-中转站2
        (6, 1, 3.0, 5, 0.9, 4.0),   # 中转站1-客户A
        (6, 2, 4.0, 6, 0.8, 5.0),   # 中转站1-客户B
        (7, 3, 3.5, 5, 0.8, 4.5),   # 中转站2-客户C
        (7, 4, 5.0, 7, 0.7, 6.0),   # 中转站2-客户D
        (6, 5, 6.0, 8, 0.6, 7.0),   # 中转站1-客户E
        (1, 2, 2.0, 3, 0.9, 3.0),   # 客户A-客户B
        (3, 4, 2.5, 4, 0.8, 3.5),   # 客户C-客户D
        (2, 5, 4.5, 6, 0.7, 5.5),   # 客户B-客户E
        (8, 6, 2.0, 3, 0.9, 2.5),   # 加油站-中转站1
        (9, 7, 1.5, 2, 0.95, 2.0),  # 检查站-中转站2
    ]
    
    for start, end, distance, time, safety, cost in logistics_routes:
        graph.add_edge(start, end, distance=distance, time=time, safety=safety, cost=cost, weight=cost)
    
    print(f"物流网络: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条路线")
    
    # 创建物流规划器
    planner = MultiObjectivePathPlanner(algorithm="MOEA/D", population_size=50, max_generations=45)
    
    # 物流约束
    planner.add_constraint(DistanceConstraint(25))  # 单次配送不超过25公里
    planner.add_constraint(TimeConstraint(40))     # 配送时间不超过40分钟
    planner.add_constraint(SafetyConstraint(0.6))  # 安全性要求
    
    # 物流权重 (成本和效率并重)
    planner.set_weights({
        'distance': 0.2,
        'time': 0.3,
        'safety': 0.2,
        'cost': 0.3  # 成本权重较高
    })
    
    # 规划到各个客户的配送路线
    warehouse = 0
    customers = [1, 2, 3, 4, 5]
    
    print(f"\n从仓库到各客户的配送路线规划:")
    
    all_results = {}
    for customer in customers:
        print(f"\n规划到 {logistics_nodes[customer]} 的路线...")
        
        result = planner.plan_path(graph, warehouse, customer)
        all_results[customer] = result
        
        if result.pareto_solutions:
            best_path = planner.get_best_solution(result, "balanced")
            
            route = " -> ".join([logistics_nodes[node] for node in best_path.nodes])
            print(f"推荐路线: {route}")
            print(f"距离: {best_path.total_distance:.1f}km, "
                  f"时间: {best_path.total_time:.1f}min, "
                  f"成本: {best_path.cost:.1f}元, "
                  f"安全: {best_path.safety_score:.2f}")
    
    # 生成配送路线可视化
    print(f"\n生成物流配送可视化...")
    visualizer = PathVisualizer()
    
    # 选择几个代表性路线进行可视化
    sample_routes = []
    for customer in [1, 3, 5]:  # 选择部分客户
        if customer in all_results and all_results[customer].pareto_solutions:
            sample_routes.append(all_results[customer].pareto_solutions[0])
    
    if sample_routes:
        visualizer.plot_path_on_graph(
            graph, sample_routes, 
            save_path='logistics_routes.png',
            title="物流配送路线优化"
        )
    
    # 生成成本效益分析
    print(f"\n=== 配送成本效益分析 ===")
    total_cost = 0
    total_time = 0
    total_distance = 0
    
    for customer, result in all_results.items():
        if result.pareto_solutions:
            path = result.pareto_solutions[0]
            total_cost += path.cost
            total_time += path.total_time
            total_distance += path.total_distance
    
    print(f"总配送成本: {total_cost:.1f} 元")
    print(f"总配送时间: {total_time:.1f} 分钟")
    print(f"总配送距离: {total_distance:.1f} 公里")
    print(f"平均单客户成本: {total_cost/len(customers):.1f} 元")


def example_travel_planning():
    """示例4: 旅游路线规划"""
    print("\n=== 示例4: 旅游路线规划 ===")
    print("场景: 规划旅游路线，平衡距离、时间、景点安全性和旅游成本")
    
    # 创建旅游网络
    graph = nx.Graph()
    
    # 旅游景点
    tourist_spots = {
        0: "酒店", 1: "博物馆", 2: "公园", 3: "购物中心", 
        4: "观景台", 5: "海滩", 6: "古建筑", 7: "美食街",
        8: "剧院", 9: "温泉", 10: "机场"
    }
    
    # 旅游路线
    tourist_routes = [
        (0, 1, 2.0, 5, 0.9, 3.0),    # 酒店-博物馆
        (0, 2, 1.5, 4, 0.8, 2.5),    # 酒店-公园
        (1, 3, 3.0, 7, 0.7, 4.0),    # 博物馆-购物中心
        (2, 4, 4.0, 8, 0.6, 5.0),    # 公园-观景台
        (3, 5, 5.0, 10, 0.5, 6.0),   # 购物中心-海滩
        (4, 6, 2.5, 6, 0.8, 3.5),    # 观景台-古建筑
        (5, 7, 3.5, 8, 0.7, 4.5),    # 海滩-美食街
        (6, 8, 4.0, 9, 0.6, 5.0),    # 古建筑-剧院
        (7, 9, 2.0, 5, 0.9, 3.0),    # 美食街-温泉
        (8, 10, 6.0, 12, 0.5, 7.0),  # 剧院-机场
        (9, 10, 4.5, 9, 0.7, 5.5),   # 温泉-机场
        (1, 6, 3.0, 7, 0.8, 4.0),    # 博物馆-古建筑
        (2, 7, 4.5, 9, 0.6, 5.5),    # 公园-美食街
    ]
    
    for start, end, distance, time, safety, cost in tourist_routes:
        graph.add_edge(start, end, distance=distance, time=time, safety=safety, cost=cost, weight=distance)
    
    print(f"旅游网络: {graph.number_of_nodes()} 个景点, {graph.number_of_edges()} 条路线")
    
    # 创建旅游规划器
    planner = MultiObjectivePathPlanner(algorithm="NSGA-II", population_size=60, max_generations=50)
    
    # 旅游约束
    planner.add_constraint(DistanceConstraint(20))  # 单日行程不超过20公里
    planner.add_constraint(TimeConstraint(60))     # 单日时间不超过60分钟
    planner.add_constraint(SafetyConstraint(0.6))  # 安全性要求
    
    # 旅游权重 (平衡各目标)
    planner.set_weights({
        'distance': 0.2,
        'time': 0.2,
        'safety': 0.3,  # 安全性权重较高
        'cost': 0.3     # 成本考虑
    })
    
    # 规划不同的旅游路线
    hotel = 0
    destinations = [5, 6, 8]  # 海滩、古建筑、剧院
    
    print(f"\n从酒店出发的旅游路线规划:")
    
    for destination in destinations:
        print(f"\n规划到 {tourist_spots[destination]} 的路线...")
        
        result = planner.plan_path(graph, hotel, destination)
        
        if result.pareto_solutions:
            print(f"找到 {len(result.pareto_solutions)} 个旅游方案:")
            
            for i, path in enumerate(result.pareto_solutions):
                route = " -> ".join([tourist_spots[node] for node in path.nodes])
                print(f"\n方案 {i+1}: {route}")
                print(f"  距离: {path.total_distance:.1f} 公里")
                print(f"  游览时间: {path.total_time:.1f} 分钟")
                print(f"  安全性: {path.safety_score:.2f}")
                print(f"  预估费用: {path.cost:.1f} 元")
                
                # 计算性价比
                value_score = (path.safety_score * 2 + (10 - path.cost/2)) / (path.total_distance + 1)
                print(f"  性价比评分: {value_score:.2f}")
            
            # 推荐最佳方案
            recommended = planner.get_best_solution(result, "balanced")
            route = " -> ".join([tourist_spots[node] for node in recommended.nodes])
            print(f"\n推荐方案: {route}")
            print(f"推荐理由: 综合考虑距离、时间、安全性和成本的平衡方案")
    
    # 生成旅游路线可视化
    print(f"\n生成旅游路线可视化...")
    visualizer = PathVisualizer()
    
    # 显示所有景点的连接情况
    visualizer.plot_path_on_graph(
        graph, [], 
        save_path='tourist_network.png',
        title="旅游景点网络图"
    )


def example_comparative_analysis():
    """示例5: 算法比较分析"""
    print("\n=== 示例5: 算法比较分析 ===")
    print("比较不同多目标优化算法在路径规划中的表现")
    
    # 创建测试图
    graph = create_sample_graph(num_nodes=20, edge_probability=0.4)
    print(f"测试图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    
    algorithms = ["NSGA-II", "MOEA/D"]
    results = {}
    
    # 统一约束条件
    constraints = [
        DistanceConstraint(40),
        TimeConstraint(50),
        SafetyConstraint(0.4)
    ]
    
    # 统一权重
    weights = {
        'distance': 0.25,
        'time': 0.25, 
        'safety': 0.25,
        'cost': 0.25
    }
    
    print(f"\n开始算法比较测试...")
    
    for algorithm in algorithms:
        print(f"\n测试 {algorithm} 算法:")
        
        # 创建规划器
        planner = MultiObjectivePathPlanner(
            algorithm=algorithm,
            population_size=50,
            max_generations=40
        )
        
        # 设置约束和权重
        for constraint in constraints:
            planner.add_constraint(constraint)
        planner.set_weights(weights)
        
        # 执行优化
        result = planner.plan_path(graph, 0, graph.number_of_nodes() - 1)
        results[algorithm] = result
        
        # 输出结果
        print(f"  解的数量: {len(result.pareto_solutions)}")
        print(f"  执行时间: {result.execution_time:.2f} 秒")
        print(f"  迭代次数: {result.iterations}")
        
        if result.pareto_solutions:
            # 分析解集质量
            analysis = planner.analyze_solutions(result)
            
            print(f"  解集多样性: {analysis['solution_diversity']['mean_distance']:.2f}")
            
            # 显示目标范围
            for obj, stats in analysis['objective_ranges'].items():
                print(f"    {obj}: [{stats['min']:.1f}, {stats['max']:.1f}] "
                      f"(均值: {stats['mean']:.1f})")
    
    # 生成算法比较可视化
    print(f"\n生成算法比较可视化...")
    
    visualizer = PathVisualizer()
    
    # 比较帕累托前沿
    if all(results[alg].pareto_solutions for alg in algorithms):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        objectives = ['distance', 'time', 'safety', 'cost']
        
        for i, obj in enumerate(objectives):
            ax = axes[i//2, i%2]
            
            for algorithm, color in zip(algorithms, ['blue', 'red']):
                if results[algorithm].pareto_solutions:
                    values = [path.objectives.get(obj, 0) for path in results[algorithm].pareto_solutions]
                    ax.scatter(values, [algorithm]*len(values), 
                             c=color, alpha=0.7, label=algorithm, s=50)
            
            ax.set_xlabel(obj)
            ax.set_title(f'{obj} 分布比较')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 输出总结
    print(f"\n=== 算法比较总结 ===")
    print("算法\t\t解数\t时间(秒)\t迭代数\t多样性")
    for algorithm in algorithms:
        if algorithm in results:
            result = results[algorithm]
            diversity = planner.analyze_solutions(result)['solution_diversity']['mean_distance']
            print(f"{algorithm:<8}\t{len(result.pareto_solutions)}\t"
                  f"{result.execution_time:.2f}\t{result.iterations}\t{diversity:.2f}")


def main():
    """主函数 - 运行所有示例"""
    print("多目标路径规划系统示例演示")
    print("=" * 50)
    
    # 设置随机种子
    np.random.seed(42)
    
    try:
        # 运行各种示例
        example_city_navigation()
        example_emergency_routing()
        example_logistics_optimization()
        example_travel_planning()
        example_comparative_analysis()
        
        print(f"\n" + "=" * 50)
        print("所有示例演示完成！")
        print(f"生成的文件:")
        print(f"  - city_navigation_paths.png: 城市导航路径图")
        print(f"  - city_navigation_pareto.png: 导航帕累托前沿")
        print(f"  - logistics_routes.png: 物流配送路线图")
        print(f"  - tourist_network.png: 旅游网络图")
        print(f"  - algorithm_comparison.png: 算法比较图")
        
    except Exception as e:
        print(f"示例运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()