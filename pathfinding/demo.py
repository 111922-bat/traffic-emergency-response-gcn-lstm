#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标路径规划系统演示脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_objective_planner import (
    MultiObjectivePathPlanner, 
    DistanceConstraint, TimeConstraint, SafetyConstraint,
    create_sample_graph, PathVisualizer
)
import numpy as np


def demo_basic_usage():
    """演示基本使用方法"""
    print("=== 基本使用演示 ===")
    
    # 创建测试图
    graph = create_sample_graph(num_nodes=10, edge_probability=0.4)
    print(f"创建了包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边的测试图")
    
    # 创建规划器
    planner = MultiObjectivePathPlanner(algorithm="NSGA-II", population_size=30, max_generations=20)
    
    # 添加约束
    planner.add_constraint(DistanceConstraint(25))
    planner.add_constraint(TimeConstraint(35))
    planner.add_constraint(SafetyConstraint(0.4))
    
    # 设置权重
    planner.set_weights({
        'distance': 0.3,
        'time': 0.3,
        'safety': 0.2,
        'cost': 0.2
    })
    
    # 规划路径
    result = planner.plan_path(graph, 0, graph.number_of_nodes() - 1)
    
    print(f"找到 {len(result.pareto_solutions)} 个帕累托最优解")
    
    if result.pareto_solutions:
        print("\n前3个最优解:")
        for i, path in enumerate(result.pareto_solutions[:3]):
            print(f"解 {i+1}: 距离={path.total_distance:.1f}, 时间={path.total_time:.1f}, "
                  f"安全性={path.safety_score:.2f}, 成本={path.cost:.1f}")
        
        # 获取不同偏好的最佳解
        print("\n不同偏好下的最佳解:")
        preferences = ['balanced', 'shortest', 'fastest', 'safest', 'cheapest']
        for pref in preferences:
            best_path = planner.get_best_solution(result, pref)
            if best_path:
                print(f"{pref:>8}: 距离={best_path.total_distance:.1f}, "
                      f"时间={best_path.total_time:.1f}, 安全性={best_path.safety_score:.2f}")
    
    return result


def demo_algorithm_comparison():
    """演示算法比较"""
    print("\n=== 算法比较演示 ===")
    
    # 创建测试图
    graph = create_sample_graph(num_nodes=12, edge_probability=0.4)
    
    algorithms = ["NSGA-II", "MOEA/D"]
    results = {}
    
    for algorithm in algorithms:
        print(f"\n测试 {algorithm} 算法...")
        
        planner = MultiObjectivePathPlanner(
            algorithm=algorithm, 
            population_size=25, 
            max_generations=15
        )
        
        # 添加约束
        planner.add_constraint(DistanceConstraint(30))
        planner.add_constraint(TimeConstraint(40))
        planner.add_constraint(SafetyConstraint(0.3))
        
        # 设置权重
        planner.set_weights({
            'distance': 0.25,
            'time': 0.25,
            'safety': 0.25,
            'cost': 0.25
        })
        
        result = planner.plan_path(graph, 0, graph.number_of_nodes() - 1)
        results[algorithm] = result
        
        print(f"  解的数量: {len(result.pareto_solutions)}")
        print(f"  执行时间: {result.execution_time:.2f} 秒")
        print(f"  迭代次数: {result.iterations}")
    
    # 算法比较总结
    print(f"\n=== 算法比较总结 ===")
    print("算法\t\t解数\t时间(秒)\t迭代数")
    for algorithm in algorithms:
        if algorithm in results:
            result = results[algorithm]
            print(f"{algorithm:<8}\t{len(result.pareto_solutions)}\t"
                  f"{result.execution_time:.2f}\t\t{result.iterations}")
    
    return results


def demo_constraints():
    """演示约束条件影响"""
    print("\n=== 约束条件演示 ===")
    
    # 创建测试图
    graph = create_sample_graph(num_nodes=8, edge_probability=0.5)
    
    # 不同的约束场景
    scenarios = [
        {
            "name": "宽松约束",
            "constraints": [
                DistanceConstraint(40),
                TimeConstraint(50),
                SafetyConstraint(0.2)
            ]
        },
        {
            "name": "中等约束",
            "constraints": [
                DistanceConstraint(20),
                TimeConstraint(30),
                SafetyConstraint(0.5)
            ]
        },
        {
            "name": "严格约束",
            "constraints": [
                DistanceConstraint(10),
                TimeConstraint(15),
                SafetyConstraint(0.7)
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n测试场景: {scenario['name']}")
        
        planner = MultiObjectivePathPlanner(algorithm="NSGA-II", population_size=20, max_generations=15)
        
        for constraint in scenario['constraints']:
            planner.add_constraint(constraint)
        
        try:
            result = planner.plan_path(graph, 0, graph.number_of_nodes() - 1)
            print(f"  找到 {len(result.pareto_solutions)} 个可行解")
            
            if result.pareto_solutions:
                best_path = planner.get_best_solution(result, "balanced")
                print(f"  最佳解: 距离={best_path.total_distance:.1f}, "
                      f"时间={best_path.total_time:.1f}, "
                      f"安全性={best_path.safety_score:.2f}")
        except Exception as e:
            print(f"  错误: {e}")


def demo_visualization():
    """演示可视化功能"""
    print("\n=== 可视化演示 ===")
    
    # 创建测试图
    graph = create_sample_graph(num_nodes=10, edge_probability=0.4)
    
    planner = MultiObjectivePathPlanner(algorithm="NSGA-II", population_size=25, max_generations=15)
    
    # 添加约束
    planner.add_constraint(DistanceConstraint(25))
    planner.add_constraint(TimeConstraint(35))
    planner.add_constraint(SafetyConstraint(0.3))
    
    result = planner.plan_path(graph, 0, graph.number_of_nodes() - 1)
    
    if result.pareto_solutions:
        print(f"生成了 {len(result.pareto_solutions)} 个解的可视化")
        
        # 创建可视化器
        visualizer = PathVisualizer()
        
        print("生成帕累托前沿图...")
        try:
            visualizer.plot_pareto_frontier(result, save_path='demo_pareto_frontier.png')
            print("  ✓ 帕累托前沿图已保存")
        except Exception as e:
            print(f"  ✗ 帕累托前沿图生成失败: {e}")
        
        print("生成路径图...")
        try:
            visualizer.plot_path_on_graph(graph, result.pareto_solutions[:3], 
                                        save_path='demo_paths.png')
            print("  ✓ 路径图已保存")
        except Exception as e:
            print(f"  ✗ 路径图生成失败: {e}")
        
        print("生成收敛曲线...")
        try:
            visualizer.plot_convergence(result, 'distance')
            print("  ✓ 收敛曲线已生成")
        except Exception as e:
            print(f"  ✗ 收敛曲线生成失败: {e}")
    else:
        print("未找到可行解，无法生成可视化")


def main():
    """主演示函数"""
    print("多目标路径规划系统演示")
    print("=" * 50)
    
    # 设置随机种子
    np.random.seed(42)
    
    try:
        # 基本使用演示
        demo_basic_usage()
        
        # 算法比较演示
        demo_algorithm_comparison()
        
        # 约束条件演示
        demo_constraints()
        
        # 可视化演示
        demo_visualization()
        
        print("\n" + "=" * 50)
        print("演示完成！")
        print("\n生成的文件:")
        print("  - demo_pareto_frontier.png: 帕累托前沿图")
        print("  - demo_paths.png: 路径图")
        
        print("\n主要功能总结:")
        print("✓ 多目标路径规划 (距离、时间、安全性、成本)")
        print("✓ 帕累托最优解计算和排序")
        print("✓ 约束条件处理 (距离、时间、安全性)")
        print("✓ 权重调整和偏好设置")
        print("✓ 多种优化算法 (NSGA-II、MOEA/D)")
        print("✓ 解集可视化和分析")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()