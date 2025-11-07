"""
应急建议生成器使用示例
展示如何使用应急建议生成器的各种功能

作者：TrafficAI Team
日期：2025-11-05
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.emergency_advisor import (
    EmergencyAdvisor, EmergencyType, EmergencyScenario, Priority,
    OptimizationObjective, create_sample_road_network, create_sample_prediction_data
)


def basic_usage_example():
    """基础使用示例"""
    print("=" * 60)
    print("应急建议生成器基础使用示例")
    print("=" * 60)
    
    # 1. 创建应急建议生成器
    print("\n1. 创建应急建议生成器...")
    advisor = EmergencyAdvisor()
    print("✓ 应急建议生成器创建成功")
    
    # 2. 加载路网数据
    print("\n2. 加载路网数据...")
    road_network = create_sample_road_network()
    advisor.load_road_network_data(road_network)
    print(f"✓ 成功加载 {len(road_network)} 个路段")
    
    # 3. 加载交通信号灯数据（示例）
    print("\n3. 加载交通信号灯数据...")
    traffic_lights = {
        "light_001": {
            "location": (39.9042, 116.4074),
            "current_phase": "green",
            "cycle_time": 120,
            "green_time": {"green": 60, "yellow": 3, "red": 57},
            "coordination_group": "group_A"
        },
        "light_002": {
            "location": (39.9052, 116.4084),
            "current_phase": "red",
            "cycle_time": 90,
            "green_time": {"green": 45, "yellow": 3, "red": 42},
            "coordination_group": "group_A"
        }
    }
    advisor.load_traffic_light_data(traffic_lights)
    print(f"✓ 成功加载 {len(traffic_lights)} 个信号灯")
    
    # 4. 生成应急建议
    print("\n4. 生成应急建议...")
    prediction_data = create_sample_prediction_data()
    suggestions = advisor.analyze_congestion_prediction(prediction_data)
    
    print(f"✓ 生成了 {len(suggestions)} 条应急建议:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion.title}")
        print(f"      类型: {suggestion.type.value}")
        print(f"      场景: {suggestion.scenario.value}")
        print(f"      优先级: {suggestion.priority.name}")
        print(f"      成本: {suggestion.implementation_cost:,.0f} 元")
        print(f"      预期收益: {suggestion.expected_benefit:,.0f} 元")
        print(f"      有效性: {suggestion.effectiveness_score:.2f}")
        print(f"      实施时间: {suggestion.implementation_time} 分钟")
        print()
    
    return advisor, suggestions


def advanced_optimization_example():
    """高级优化示例"""
    print("=" * 60)
    print("应急建议生成器高级优化示例")
    print("=" * 60)
    
    advisor = EmergencyAdvisor()
    road_network = create_sample_road_network()
    advisor.load_road_network_data(road_network)
    
    # 生成基础建议
    prediction_data = create_sample_prediction_data()
    suggestions = advisor.analyze_congestion_prediction(prediction_data)
    
    print(f"\n生成了 {len(suggestions)} 条基础建议")
    
    # 1. 自定义优化目标
    print("\n1. 设置自定义优化目标...")
    custom_objectives = OptimizationObjective(
        minimize_time=1.5,      # 更加重视时间优化
        minimize_distance=0.5,  # 距离权重较低
        minimize_cost=0.8,      # 成本权重适中
        maximize_safety=1.2,    # 高度重视安全性
        minimize_congestion=1.0 # 拥堵减少权重标准
    )
    
    recommendations = advisor.get_optimization_recommendations(custom_objectives)
    
    print("✓ 自定义优化目标设置完成")
    print(f"   时间权重: {custom_objectives.minimize_time}")
    print(f"   安全性权重: {custom_objectives.maximize_safety}")
    
    # 2. 性能分析
    print("\n2. 性能分析结果...")
    performance = recommendations["current_performance"]
    print(f"   整体有效性: {performance['overall_effectiveness']:.3f}")
    print(f"   成本效益比: {performance['benefit_cost_ratio']:.3f}")
    print(f"   平均实施时间: {performance['average_implementation_time']:.1f} 分钟")
    
    # 3. 优化建议
    print("\n3. 系统优化建议...")
    optimization_suggestions = recommendations["optimization_suggestions"]
    if optimization_suggestions:
        for i, suggestion in enumerate(optimization_suggestions, 1):
            print(f"   {i}. {suggestion}")
    else:
        print("   当前系统运行良好，暂无优化建议")
    
    # 4. 资源分配优化
    print("\n4. 资源分配优化...")
    resource_allocation = recommendations["resource_allocation"]
    cost_allocation = resource_allocation["cost_allocation"]
    print(f"   高优先级成本占比: {cost_allocation['high_priority']:.1%}")
    print(f"   中优先级成本占比: {cost_allocation['medium_priority']:.1%}")
    print(f"   低优先级成本占比: {cost_allocation['low_priority']:.1%}")
    
    # 5. 实施优先级排序
    print("\n5. 实施优先级排序...")
    priorities = recommendations["implementation_priorities"]
    print("   建议实施顺序:")
    for i, priority in enumerate(priorities[:5], 1):  # 显示前5个
        print(f"   {i}. {priority['title']}")
        print(f"      优先级评分: {priority['priority_score']:.3f}")
        print(f"      实施时间: {priority['implementation_time']} 分钟")
        print(f"      成本: {priority['cost']:,.0f} 元")
        print(f"      有效性: {priority['effectiveness']:.3f}")
        print()
    
    return advisor


def realtime_update_example():
    """实时更新示例"""
    print("=" * 60)
    print("应急建议生成器实时更新示例")
    print("=" * 60)
    
    advisor = EmergencyAdvisor()
    road_network = create_sample_road_network()
    advisor.load_road_network_data(road_network)
    
    # 初始预测数据
    initial_prediction = create_sample_prediction_data()
    initial_suggestions = advisor.analyze_congestion_prediction(initial_prediction)
    
    print(f"初始生成了 {len(initial_suggestions)} 条建议")
    print(f"当前活跃建议数量: {len(advisor.active_suggestions)}")
    
    # 模拟实时更新
    print("\n模拟实时交通状况变化...")
    
    for update_round in range(3):
        print(f"\n第 {update_round + 1} 次更新:")
        
        # 模拟新的交通状况
        new_prediction = {
            "hotspots": [
                {
                    "id": f"hotspot_update_{update_round}",
                    "location": (39.907 + update_round * 0.001, 116.41 + update_round * 0.001),
                    "affected_segments": ["segment_003"],
                    "severity": np.random.randint(1, 4),
                    "accident_indicators": np.random.choice([True, False]),
                    "weather_impact": np.random.choice([True, False]),
                    "event_related": np.random.choice([True, False]),
                    "data_quality": np.random.uniform(0.7, 1.0),
                    "prediction_accuracy": np.random.uniform(0.6, 0.9)
                }
            ],
            "propagation_patterns": [],
            "severity_levels": [np.random.randint(1, 4)]
        }
        
        # 执行实时更新
        updated_suggestions = advisor.update_suggestions_realtime(new_prediction)
        
        print(f"  新增建议: {len(updated_suggestions)} 条")
        print(f"  当前活跃建议: {len(advisor.active_suggestions)} 条")
        
        # 显示新增的重要建议
        if updated_suggestions:
            print("  新增建议详情:")
            for suggestion in updated_suggestions[:3]:  # 显示前3个
                print(f"    - {suggestion.title} (优先级: {suggestion.priority.name})")
        
        # 模拟时间流逝
        time.sleep(0.1)
    
    # 最终状态
    print(f"\n最终状态:")
    print(f"  总活跃建议: {len(advisor.active_suggestions)}")
    print(f"  建议历史记录: {len(advisor.suggestion_history)}")
    
    return advisor


def scenario_specific_example():
    """特定场景示例"""
    print("=" * 60)
    print("不同应急场景专项示例")
    print("=" * 60)
    
    advisor = EmergencyAdvisor()
    road_network = create_sample_road_network()
    advisor.load_road_network_data(road_network)
    
    # 定义不同场景的预测数据
    scenarios = {
        "交通事故": {
            "hotspots": [{
                "id": "accident_hotspot",
                "location": (39.9052, 116.4084),
                "affected_segments": ["segment_001", "segment_002"],
                "severity": 3,
                "accident_indicators": True,
                "weather_impact": False,
                "event_related": False,
                "data_quality": 0.9,
                "prediction_accuracy": 0.85
            }],
            "propagation_patterns": [],
            "severity_levels": [3]
        },
        "恶劣天气": {
            "hotspots": [{
                "id": "weather_hotspot",
                "location": (39.9062, 116.4094),
                "affected_segments": ["segment_003"],
                "severity": 2,
                "accident_indicators": False,
                "weather_impact": True,
                "event_related": False,
                "data_quality": 0.85,
                "prediction_accuracy": 0.75
            }],
            "propagation_patterns": [],
            "severity_levels": [2]
        },
        "大型活动": {
            "hotspots": [{
                "id": "event_hotspot",
                "location": (39.9042, 116.4074),
                "affected_segments": ["segment_001", "segment_003"],
                "severity": 2,
                "accident_indicators": False,
                "weather_impact": False,
                "event_related": True,
                "data_quality": 0.8,
                "prediction_accuracy": 0.7
            }],
            "propagation_patterns": [],
            "severity_levels": [2]
        }
    }
    
    # 为每个场景生成建议
    for scenario_name, prediction_data in scenarios.items():
        print(f"\n{scenario_name}场景应急建议:")
        print("-" * 40)
        
        suggestions = advisor.analyze_congestion_prediction(prediction_data)
        
        # 按类型分组显示
        suggestions_by_type = {}
        for suggestion in suggestions:
            if suggestion.type not in suggestions_by_type:
                suggestions_by_type[suggestion.type] = []
            suggestions_by_type[suggestion.type].append(suggestion)
        
        for emergency_type, type_suggestions in suggestions_by_type.items():
            print(f"\n{emergency_type.value}:")
            for suggestion in type_suggestions:
                print(f"  • {suggestion.title}")
                print(f"    优先级: {suggestion.priority.name}")
                print(f"    成本: {suggestion.implementation_cost:,.0f} 元")
                print(f"    收益: {suggestion.expected_benefit:,.0f} 元")
                print(f"    有效性: {suggestion.effectiveness_score:.2f}")
                print(f"    实施时间: {suggestion.implementation_time} 分钟")
                
                # 显示资源需求
                if suggestion.resource_requirements:
                    print(f"    资源需求: {suggestion.resource_requirements}")
                
                # 显示风险因素
                if suggestion.risk_factors:
                    print(f"    风险因素: {', '.join(suggestion.risk_factors)}")
                print()
    
    return advisor


def report_export_example():
    """报告导出示例"""
    print("=" * 60)
    print("应急建议报告导出示例")
    print("=" * 60)
    
    advisor = EmergencyAdvisor()
    road_network = create_sample_road_network()
    advisor.load_road_network_data(road_network)
    
    # 生成建议
    prediction_data = create_sample_prediction_data()
    advisor.analyze_congestion_prediction(prediction_data)
    
    # 1. 导出JSON报告
    print("\n1. 导出JSON格式报告...")
    json_report = advisor.export_suggestions_report("json")
    
    # 解析并显示报告摘要
    try:
        report_data = json.loads(json_report)
        summary = report_data["summary"]
        
        print("✓ JSON报告导出成功")
        print(f"   报告生成时间: {report_data['timestamp']}")
        print(f"   建议总数: {summary['total_suggestions']}")
        print(f"   平均有效性: {summary['average_effectiveness']:.3f}")
        print(f"   总成本: {summary['total_cost']:,.0f} 元")
        print(f"   总预期收益: {summary['total_expected_benefit']:,.0f} 元")
        print(f"   成本效益比: {summary['total_expected_benefit']/max(summary['total_cost'], 1):.2f}")
        
        # 保存报告到文件
        with open("/workspace/code/services/emergency_report.json", "w", encoding="utf-8") as f:
            f.write(json_report)
        print("   报告已保存到: emergency_report.json")
        
    except json.JSONDecodeError:
        print("✗ JSON报告格式错误")
    
    # 2. 导出CSV报告
    print("\n2. 导出CSV格式报告...")
    csv_report = advisor.export_suggestions_report("csv")
    
    # 显示CSV报告前几行
    lines = csv_report.split('\n')
    print("✓ CSV报告导出成功")
    print(f"   报告行数: {len(lines)}")
    print("   报告预览:")
    for i, line in enumerate(lines[:6]):  # 显示前6行
        print(f"   {i+1}: {line}")
    
    # 保存CSV报告
    with open("/workspace/code/services/emergency_report.csv", "w", encoding="utf-8") as f:
        f.write(csv_report)
    print("   报告已保存到: emergency_report.csv")
    
    # 3. 系统状态报告
    print("\n3. 系统状态报告...")
    status = advisor.get_system_status()
    
    print("✓ 系统状态:")
    print(f"   活跃建议数量: {status['active_suggestions_count']}")
    print(f"   建议历史记录: {status['suggestion_history_size']}")
    print(f"   路网路段数: {status['road_network_size']}")
    print(f"   信号灯数量: {status['traffic_lights_count']}")
    print(f"   最后更新时间: {datetime.fromtimestamp(status['last_update_time']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    return advisor


def performance_monitoring_example():
    """性能监控示例"""
    print("=" * 60)
    print("应急建议生成器性能监控示例")
    print("=" * 60)
    
    advisor = EmergencyAdvisor()
    
    # 创建较大的测试数据集
    print("\n创建大规模测试数据...")
    large_road_network = {}
    for i in range(50):
        large_road_network[f"segment_{i:03d}"] = {
            "start_node": f"node_{i}",
            "end_node": f"node_{i+1}",
            "length": np.random.uniform(0.5, 5.0),
            "lanes": np.random.randint(1, 6),
            "current_speed": np.random.uniform(20, 80),
            "free_flow_speed": np.random.uniform(60, 100),
            "capacity": np.random.randint(1000, 5000),
            "current_volume": np.random.randint(500, 4000),
            "congestion_level": np.random.randint(0, 4),
            "coordinates": (39.9 + np.random.uniform(-0.1, 0.1), 
                          116.4 + np.random.uniform(-0.1, 0.1))
        }
    
    advisor.load_road_network_data(large_road_network)
    print(f"✓ 加载了 {len(large_road_network)} 个路段")
    
    # 性能测试
    print("\n执行性能测试...")
    
    # 创建多个预测热点
    prediction_data = {
        "hotspots": [
            {
                "id": f"hotspot_{i}",
                "location": (39.9 + np.random.uniform(-0.1, 0.1), 
                           116.4 + np.random.uniform(-0.1, 0.1)),
                "affected_segments": [f"segment_{j:03d}" for j in range(i*2, min((i+1)*2, len(large_road_network)))],
                "severity": np.random.randint(1, 4),
                "accident_indicators": np.random.choice([True, False]),
                "weather_impact": np.random.choice([True, False]),
                "event_related": np.random.choice([True, False]),
                "data_quality": np.random.uniform(0.7, 1.0),
                "prediction_accuracy": np.random.uniform(0.6, 0.9)
            }
            for i in range(10)
        ],
        "propagation_patterns": [],
        "severity_levels": [np.random.randint(1, 4) for _ in range(10)]
    }
    
    # 测量处理时间
    start_time = time.time()
    suggestions = advisor.analyze_congestion_prediction(prediction_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    print(f"✓ 处理完成:")
    print(f"   处理时间: {processing_time:.3f} 秒")
    print(f"   生成建议数: {len(suggestions)}")
    print(f"   平均每条建议时间: {processing_time/len(suggestions):.4f} 秒")
    print(f"   每秒处理建议数: {len(suggestions)/processing_time:.1f} 条/秒")
    
    # 内存使用情况
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"   内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        print("   内存监控需要 psutil 库")
    
    # 建议质量分析
    print(f"\n建议质量分析:")
    if suggestions:
        effectiveness_scores = [s.effectiveness_score for s in suggestions]
        confidence_levels = [s.confidence_level for s in suggestions]
        costs = [s.implementation_cost for s in suggestions]
        benefits = [s.expected_benefit for s in suggestions]
        
        print(f"   有效性评分: 平均 {np.mean(effectiveness_scores):.3f}, 范围 [{np.min(effectiveness_scores):.3f}, {np.max(effectiveness_scores):.3f}]")
        print(f"   信心度: 平均 {np.mean(confidence_levels):.3f}, 范围 [{np.min(confidence_levels):.3f}, {np.max(confidence_levels):.3f}]")
        print(f"   成本: 平均 {np.mean(costs):,.0f} 元, 总计 {np.sum(costs):,.0f} 元")
        print(f"   收益: 平均 {np.mean(benefits):,.0f} 元, 总计 {np.sum(benefits):,.0f} 元")
        print(f"   整体效益比: {np.sum(benefits)/max(np.sum(costs), 1):.2f}")
    
    return advisor


def main():
    """主函数"""
    print("应急建议生成器完整示例")
    print("作者: TrafficAI Team")
    print("日期: 2025-11-05")
    print("=" * 60)
    
    try:
        # 1. 基础使用示例
        advisor1, suggestions1 = basic_usage_example()
        
        # 2. 高级优化示例
        advisor2 = advanced_optimization_example()
        
        # 3. 实时更新示例
        advisor3 = realtime_update_example()
        
        # 4. 特定场景示例
        advisor4 = scenario_specific_example()
        
        # 5. 报告导出示例
        advisor5 = report_export_example()
        
        # 6. 性能监控示例
        advisor6 = performance_monitoring_example()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("应急建议生成器功能验证成功")
        print("=" * 60)
        
        # 清理生成的报告文件
        import os
        report_files = [
            "/workspace/code/services/emergency_report.json",
            "/workspace/code/services/emergency_report.csv"
        ]
        
        print("\n清理临时文件...")
        for file_path in report_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✓ 已删除: {file_path}")
        
    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()