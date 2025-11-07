"""
最短路径算法模块使用示例
演示各种功能和实际应用场景
"""

import sys
import os
import time
import random
from typing import List, Dict, Any

# 添加模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathfinding.shortest_path import (
    Graph, Node, Edge, RoadType, ShortestPathEngine, 
    ShortestPathAlgorithms, PathResult
)


def create_city_road_network() -> Graph:
    """创建城市道路网络示例"""
    print("🏙️  创建城市道路网络...")
    
    graph = Graph()
    
    # 主要区域节点
    locations = {
        "市中心": Node("市中心", 0, 0, elevation=10),
        "火车站": Node("火车站", 2, 1, elevation=8),
        "机场": Node("机场", 8, 2, elevation=5),
        "港口": Node("港口", 1, -3, elevation=2),
        "工业区": Node("工业区", -2, 2, elevation=15),
        "住宅区A": Node("住宅区A", -3, -1, elevation=12),
        "住宅区B": Node("住宅区B", 3, -2, elevation=8),
        "商业区": Node("商业区", 1, 1, elevation=11),
        "大学": Node("大学", -1, 3, elevation=18),
        "医院": Node("医院", 2, -1, elevation=9),
        "公园": Node("公园", -2, 0, elevation=13),
        "政府大楼": Node("政府大楼", 0, 2, elevation=14)
    }
    
    for node in locations.values():
        graph.add_node(node)
    
    # 道路网络
    roads = [
        # 高速公路
        Edge("市中心", "机场", 15.0, RoadType.HIGHWAY, speed_limit=100, is_toll=True, toll_cost=5.0),
        Edge("市中心", "火车站", 8.0, RoadType.HIGHWAY, speed_limit=80, is_toll=True, toll_cost=3.0),
        Edge("机场", "火车站", 12.0, RoadType.HIGHWAY, speed_limit=90, is_toll=True, toll_cost=4.0),
        
        # 主干道
        Edge("市中心", "商业区", 2.0, RoadType.ARTERIAL, speed_limit=60),
        Edge("商业区", "火车站", 3.0, RoadType.ARTERIAL, speed_limit=60),
        Edge("市中心", "政府大楼", 3.0, RoadType.ARTERIAL, speed_limit=60),
        Edge("政府大楼", "大学", 4.0, RoadType.ARTERIAL, speed_limit=60),
        Edge("市中心", "医院", 4.0, RoadType.ARTERIAL, speed_limit=60),
        Edge("医院", "住宅区B", 3.0, RoadType.ARTERIAL, speed_limit=50),
        
        # 集散道路
        Edge("火车站", "港口", 10.0, RoadType.COLLECTOR, speed_limit=50),
        Edge("市中心", "工业区", 8.0, RoadType.COLLECTOR, speed_limit=50),
        Edge("工业区", "大学", 6.0, RoadType.COLLECTOR, speed_limit=50),
        Edge("住宅区A", "公园", 2.0, RoadType.COLLECTOR, speed_limit=40),
        Edge("公园", "市中心", 3.0, RoadType.COLLECTOR, speed_limit=40),
        
        # 地方道路
        Edge("住宅区A", "市中心", 5.0, RoadType.LOCAL, speed_limit=30),
        Edge("住宅区A", "住宅区B", 7.0, RoadType.LOCAL, speed_limit=30),
        Edge("商业区", "公园", 4.0, RoadType.LOCAL, speed_limit=30),
        Edge("大学", "工业区", 5.0, RoadType.LOCAL, speed_limit=30),
        Edge("医院", "商业区", 2.0, RoadType.LOCAL, speed_limit=30),
        
        # 桥梁和特殊道路
        Edge("港口", "住宅区B", 6.0, RoadType.BRIDGE, speed_limit=40, max_height=4.5, max_weight=20.0),
        Edge("市中心", "住宅区A", 4.0, RoadType.TUNNEL, speed_limit=50, max_height=3.8),
        
        # 匝道
        Edge("市中心", "机场", 18.0, RoadType.RAMP, speed_limit=40)  # 备用路线
    ]
    
    for road in roads:
        graph.add_edge(road)
    
    print(f"   ✅ 创建完成: {len(locations)}个节点, {len(roads)}条道路")
    return graph


def demonstrate_basic_pathfinding():
    """演示基本路径查找功能"""
    print("\n🛣️  基本路径查找演示")
    print("-" * 40)
    
    graph = create_city_road_network()
    engine = ShortestPathEngine()
    engine.load_graph(graph)
    
    # 测试用例
    test_cases = [
        ("市中心", "机场", "前往机场"),
        ("住宅区A", "大学", "从住宅区到大学"),
        ("港口", "政府大楼", "从港口到政府大楼"),
        ("医院", "机场", "从医院到机场")
    ]
    
    for start, end, description in test_cases:
        print(f"\n📍 {description}")
        print(f"   起点: {start} → 终点: {end}")
        
        # 测试不同算法
        algorithms = ["dijkstra", "astar", "floyd"]
        
        for algo in algorithms:
            start_time = time.time()
            result = engine.find_shortest_path(start, end, algo)
            end_time = time.time()
            
            if result.is_valid:
                print(f"   {algo.upper():>8}: {' → '.join(result.path)}")
                print(f"           距离: {result.total_distance:.1f}km")
                print(f"           时间: {result.computation_time*1000:.2f}ms")
            else:
                print(f"   {algo.upper():>8}: 无法找到路径")
                if result.warnings:
                    print(f"           警告: {result.warnings}")


def demonstrate_constraint_based_routing():
    """演示基于约束的路径规划"""
    print("\n🚛 约束条件路径规划演示")
    print("-" * 40)
    
    graph = create_city_road_network()
    engine = ShortestPathEngine()
    engine.load_graph(graph)
    
    # 模拟不同类型的车辆
    vehicles = [
        {
            "name": "小型轿车",
            "height": 1.5,
            "weight": 1.5,
            "avoid_tolls": False,
            "preferred_types": {RoadType.HIGHWAY, RoadType.ARTERIAL}
        },
        {
            "name": "大型货车",
            "height": 4.0,
            "weight": 25.0,
            "avoid_tolls": True,
            "preferred_types": {RoadType.HIGHWAY, RoadType.ARTERIAL}
        },
        {
            "name": "超高车辆",
            "height": 4.2,
            "weight": 2.0,
            "avoid_tolls": False,
            "preferred_types": {RoadType.ARTERIAL, RoadType.COLLECTOR}
        }
    ]
    
    start, end = "住宅区A", "机场"
    
    for vehicle in vehicles:
        print(f"\n🚗 {vehicle['name']} ({start} → {end})")
        
        constraints = {
            'vehicle_height': vehicle['height'],
            'vehicle_weight': vehicle['weight'],
            'avoid_tolls': vehicle['avoid_tolls'],
            'preferred_road_types': vehicle['preferred_types']
        }
        
        result = engine.find_shortest_path(start, end, "dijkstra", constraints)
        
        if result.is_valid and result.constraints_satisfied:
            print(f"   路径: {' → '.join(result.path)}")
            print(f"   距离: {result.total_distance:.1f}km")
            
            # 计算预计时间
            total_time = 0
            for edge in result.edges:
                if edge.speed_limit:
                    total_time += edge.weight / edge.speed_limit * 60  # 转换为分钟
            
            print(f"   预计时间: {total_time:.1f}分钟")
            
            # 检查是否有收费道路
            toll_cost = sum(edge.toll_cost for edge in result.edges if edge.is_toll)
            if toll_cost > 0:
                print(f"   收费: {toll_cost:.1f}元")
            else:
                print(f"   收费: 无")
        else:
            print(f"   ❌ 无法找到满足约束的路径")
            if result.warnings:
                print(f"   原因: {result.warnings}")


def demonstrate_dynamic_routing():
    """演示动态路径调整"""
    print("\n🔄 动态路径调整演示")
    print("-" * 40)
    
    graph = create_city_road_network()
    engine = ShortestPathEngine()
    engine.load_graph(graph)
    
    start, end = "住宅区A", "机场"
    
    # 初始路径
    print("📊 初始交通状况:")
    result1 = engine.find_shortest_path(start, end, "dijkstra")
    if result1.is_valid:
        print(f"   路径: {' → '.join(result1.path)}")
        print(f"   距离: {result1.total_distance:.1f}km")
    
    # 模拟交通拥堵 - 增加某些道路的权重
    print("\n🚦 模拟交通拥堵:")
    congested_edges = [
        ("市中心", "机场"),
        ("市中心", "商业区"),
        ("商业区", "火车站")
    ]
    
    for from_node, to_node in congested_edges:
        engine.update_edge_weight(from_node, to_node, 
                                engine.graph.get_edge_weight(from_node, to_node) * 3)
        print(f"   {from_node} → {to_node} 权重增加3倍")
    
    # 重新计算路径
    print("\n📊 拥堵后路径:")
    result2 = engine.find_shortest_path(start, end, "dijkstra")
    if result2.is_valid:
        print(f"   路径: {' → '.join(result2.path)}")
        print(f"   距离: {result2.total_distance:.1f}km")
        
        if result1.path != result2.path:
            print("   ✅ 路径已重新规划")
        else:
            print("   ℹ️  路径保持不变")


def demonstrate_real_time_navigation():
    """演示实时导航功能"""
    print("\n🧭 实时导航演示")
    print("-" * 40)
    
    graph = create_city_road_network()
    engine = ShortestPathEngine()
    engine.load_graph(graph)
    
    # 模拟导航过程
    start = "住宅区A"
    destination = "机场"
    current = start
    
    print(f"🎯 导航开始: {start} → {destination}")
    
    # 初始路径规划
    full_path = engine.find_shortest_path(start, destination, "dijkstra")
    if not full_path.is_valid:
        print("❌ 无法找到路径")
        return
    
    path = full_path.path.copy()
    print(f"📍 规划路径: {' → '.join(path)}")
    
    # 模拟车辆行进过程中的重新规划
    steps = [
        ("到达市中心", 1),  # 模拟到达第1个节点
        ("检测到前方施工", 2),  # 模拟检测到问题
        ("重新规划路线", 3)   # 模拟重新规划
    ]
    
    for step_desc, node_index in steps:
        print(f"\n📍 {step_desc}")
        
        if node_index < len(path) - 1:
            current = path[node_index]
            remaining_dest = path[-1]
            
            # 重新规划剩余路径
            remaining_path = engine.find_shortest_path(current, remaining_dest, "dijkstra")
            
            if remaining_path.is_valid:
                # 构建完整路径
                new_full_path = path[:node_index+1] + remaining_path.path[1:]
                print(f"   新路径: {' → '.join(new_full_path)}")
                
                # 更新路径
                path = new_full_path
                
                # 模拟道路条件变化
                if "施工" in step_desc:
                    print("   🚧 模拟前方道路施工，绕行其他道路")
                    # 增加当前节点到下一节点的权重
                    if node_index < len(path) - 1:
                        next_node = path[node_index + 1]
                        current_weight = engine.graph.get_edge_weight(current, next_node)
                        if current_weight:
                            engine.update_edge_weight(current, next_node, current_weight * 5)
            else:
                print(f"   ❌ 无法从 {current} 到 {remaining_dest}")
                break
    
    print(f"\n🎉 导航完成! 最终路径: {' → '.join(path)}")


def demonstrate_algorithm_comparison():
    """演示算法性能比较"""
    print("\n⚡ 算法性能比较")
    print("-" * 40)
    
    # 创建更大的测试图
    print("📊 创建大型测试网络...")
    graph = Graph()
    
    # 创建网格状城市网络
    size = 15  # 15x15网格
    for i in range(size):
        for j in range(size):
            node_id = f"{i}_{j}"
            node = Node(node_id, i, j)
            graph.add_node(node)
    
    # 添加边（网格连接）
    for i in range(size):
        for j in range(size):
            node_id = f"{i}_{j}"
            
            # 连接到右邻居
            if j < size - 1:
                right_id = f"{i}_{j+1}"
                weight = random.uniform(1, 3)  # 随机权重
                edge = Edge(node_id, right_id, weight, RoadType.LOCAL)
                graph.add_edge(edge)
            
            # 连接到下邻居
            if i < size - 1:
                down_id = f"{i+1}_{j}"
                weight = random.uniform(1, 3)  # 随机权重
                edge = Edge(node_id, down_id, weight, RoadType.LOCAL)
                graph.add_edge(edge)
    
    engine = ShortestPathEngine()
    engine.load_graph(graph)
    
    print(f"   ✅ 网络规模: {size}x{size} = {size*size}个节点")
    
    # 测试用例
    test_pairs = [
        ("0_0", f"{size-1}_{size-1}", "对角线"),
        ("0_0", f"0_{size-1}", "水平"),
        ("0_0", f"{size-1}_0", "垂直"),
        (f"{size//2}_{size//2}", f"{size-1}_{size-1}", "中心到角落")
    ]
    
    algorithms = ["dijkstra", "astar", "floyd"]
    
    for start, end, desc in test_pairs:
        print(f"\n📍 测试: {desc} ({start} → {end})")
        
        for algo in algorithms:
            start_time = time.time()
            result = engine.find_shortest_path(start, end, algo)
            end_time = time.time()
            
            if result.is_valid:
                elapsed_ms = (end_time - start_time) * 1000
                print(f"   {algo.upper():>8}: {elapsed_ms:6.2f}ms, 距离: {result.total_distance:.2f}")
            else:
                print(f"   {algo.upper():>8}: 失败")


def demonstrate_emergency_routing():
    """演示紧急情况路径规划"""
    print("\n🚨 紧急情况路径规划演示")
    print("-" * 40)
    
    graph = create_city_road_network()
    engine = ShortestPathEngine()
    engine.load_graph(graph)
    
    # 紧急情况场景
    emergency_scenarios = [
        {
            "name": "救护车",
            "start": "住宅区B",
            "end": "医院",
            "constraints": {
                'vehicle_height': 2.5,
                'vehicle_weight': 3.0,
                'avoid_tolls': True,
                'preferred_road_types': {RoadType.HIGHWAY, RoadType.ARTERIAL, RoadType.COLLECTOR}
            },
            "priority": "最快到达医院"
        },
        {
            "name": "消防车",
            "start": "工业区",
            "end": "住宅区A",
            "constraints": {
                'vehicle_height': 3.5,
                'vehicle_weight': 15.0,
                'avoid_tolls': True,
                'preferred_road_types': {RoadType.HIGHWAY, RoadType.ARTERIAL}
            },
            "priority": "最快到达火灾现场"
        },
        {
            "name": "警车",
            "start": "市中心",
            "end": "大学",
            "constraints": {
                'vehicle_height': 2.0,
                'vehicle_weight': 2.0,
                'avoid_tolls': False,
                'preferred_road_types': {RoadType.HIGHWAY, RoadType.ARTERIAL, RoadType.COLLECTOR, RoadType.LOCAL}
            },
            "priority": "最快到达现场"
        }
    ]
    
    for scenario in emergency_scenarios:
        print(f"\n🚨 {scenario['name']} 紧急调度")
        print(f"   任务: {scenario['priority']}")
        print(f"   起点: {scenario['start']} → 终点: {scenario['end']}")
        
        start_time = time.time()
        result = engine.find_shortest_path(
            scenario['start'], 
            scenario['end'], 
            "astar",  # A*算法适合紧急情况
            scenario['constraints']
        )
        end_time = time.time()
        
        if result.is_valid:
            print(f"   ✅ 路径: {' → '.join(result.path)}")
            print(f"   📏 距离: {result.total_distance:.1f}km")
            
            # 计算预计时间（假设紧急车辆以80km/h行驶）
            estimated_time = result.total_distance / 80 * 60  # 分钟
            print(f"   ⏱️  预计时间: {estimated_time:.1f}分钟")
            print(f"   ⚡ 规划时间: {result.computation_time*1000:.2f}ms")
            
            # 检查约束满足情况
            if result.constraints_satisfied:
                print(f"   ✅ 所有约束条件满足")
            else:
                print(f"   ⚠️  部分约束条件不满足")
        else:
            print(f"   ❌ 无法找到满足条件的路径")
            if result.warnings:
                print(f"   原因: {result.warnings}")


def main():
    """主演示函数"""
    print("🚗 最短路径算法模块 - 综合演示")
    print("=" * 50)
    
    try:
        # 基本功能演示
        demonstrate_basic_pathfinding()
        
        # 约束条件演示
        demonstrate_constraint_based_routing()
        
        # 动态路径调整演示
        demonstrate_dynamic_routing()
        
        # 实时导航演示
        demonstrate_real_time_navigation()
        
        # 算法性能比较
        demonstrate_algorithm_comparison()
        
        # 紧急情况演示
        demonstrate_emergency_routing()
        
        print("\n" + "=" * 50)
        print("🎉 所有演示完成!")
        print("\n📚 使用说明:")
        print("   1. 使用 ShortestPathEngine 创建路径规划引擎")
        print("   2. 加载图数据 (Graph 对象)")
        print("   3. 调用 find_shortest_path() 进行路径查找")
        print("   4. 支持多种算法: dijkstra, astar, floyd")
        print("   5. 支持车辆约束和道路偏好设置")
        print("   6. 支持动态权重更新和实时重新规划")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()