"""
拥堵扩散预测系统使用示例

本示例展示如何使用拥堵扩散预测算法进行：
1. 基本预测功能
2. 实时预测
3. 批量预测
4. 应急控制策略
5. 结果可视化和分析

作者：TrafficAI Team
日期：2025-11-05
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from congestion_predictor import (
    CongestionPropagationPredictor, RoadSegment, CongestionLevel,
    PredictionMode, create_sample_data, visualize_prediction_results,
    evaluate_prediction_accuracy
)


def create_realistic_traffic_data(n_segments: int = 30, n_timesteps: int = 100) -> pd.DataFrame:
    """创建更真实的交通数据"""
    np.random.seed(42)  # 确保结果可重现
    
    data = []
    
    for t in range(n_timesteps):
        # 模拟日内周期性
        hour = (t * 5) % 1440  # 假设每步5分钟
        time_factor = 1.0
        
        # 早晚高峰
        if 420 <= hour <= 540:  # 7:00-9:00
            time_factor = 0.4  # 速度降低60%
        elif 1020 <= hour <= 1140:  # 17:00-19:00
            time_factor = 0.5  # 速度降低50%
        
        for i in range(n_segments):
            segment_id = f"segment_{i:03d}"
            
            # 路段特性
            length = np.random.uniform(0.5, 3.0)
            lanes = np.random.choice([2, 3, 4])
            capacity = lanes * 800 + np.random.randint(-100, 100)
            free_flow_speed = 50 + np.random.randint(0, 30)
            
            # 基础速度
            base_speed = free_flow_speed * time_factor
            
            # 添加空间相关性（相邻路段速度相关）
            if i > 0:
                prev_speed = data[-1]['current_speed'] if data else base_speed
                base_speed = 0.7 * base_speed + 0.3 * prev_speed
            
            # 添加随机扰动
            speed_noise = np.random.normal(0, 5)
            current_speed = max(5, base_speed + speed_noise)
            
            # 计算流量（基于基本图关系）
            speed_ratio = current_speed / free_flow_speed
            flow_ratio = max(0, min(1.2, speed_ratio * (2 - speed_ratio)))  # 抛物线关系
            current_flow = int(capacity * flow_ratio + np.random.normal(0, 50))
            current_flow = max(0, current_flow)
            
            # 占有率
            occupancy = min(0.95, current_flow / (capacity + 1) + np.random.uniform(0, 0.1))
            
            data.append({
                'timestamp': t,
                'segment_id': segment_id,
                'length': length,
                'lanes': lanes,
                'capacity': capacity,
                'free_flow_speed': free_flow_speed,
                'current_speed': current_speed,
                'current_flow': current_flow,
                'occupancy': occupancy
            })
    
    return pd.DataFrame(data)


def example_basic_prediction():
    """基本预测示例"""
    print("=== 基本预测示例 ===")
    
    # 配置参数
    config = {
        'input_dim': 4,
        'hidden_dim': 64,
        'output_dim': 3,
        'gcn_layers': 3,
        'lstm_layers': 2,
        'dropout': 0.1,
        'bidirectional': True,
        'fusion_weight': 0.6,
        'input_sequence_length': 12,
        'n_nodes': 20,
        'embedding_dim': 64,
        'history_length': 100,
        'bottleneck_threshold': 0.7,
        'propagation_speed_threshold': 15.0,
        'emergency_response_time': 300,
        'fundamental_diagram_type': 'triangular'
    }
    
    # 创建预测器
    predictor = CongestionPropagationPredictor(config)
    
    # 创建示例数据
    segments = create_sample_data(n_segments=20)
    
    print(f"创建了 {len(segments)} 个道路路段")
    
    # 执行预测
    print("执行拥堵扩散预测...")
    results = predictor.predict_congestion_propagation(
        segments, prediction_horizon=6, mode=PredictionMode.BATCH
    )
    
    print(f"预测完成，生成 {len(results)} 个路段的预测结果")
    
    # 分析预测结果
    print("\n预测结果分析:")
    for i, result in enumerate(results[:3]):  # 显示前3个路段的结果
        print(f"\n路段 {result.segment_id}:")
        print(f"  预测速度: {result.predicted_speeds}")
        print(f"  预测流量: {result.predicted_flows}")
        print(f"  拥堵等级: {[level.name for level in result.congestion_levels]}")
        print(f"  传播速度: {result.propagation_speed:.2f} km/h")
        print(f"  影响范围: {result.influence_range:.2f} km")
        print(f"  置信度: {result.confidence_scores}")
    
    # 识别瓶颈
    bottlenecks = predictor.identify_bottlenecks(segments)
    print(f"\n识别出 {len(bottlenecks)} 个瓶颈路段: {bottlenecks}")
    
    return predictor, results, bottlenecks


def example_real_time_prediction():
    """实时预测示例"""
    print("\n=== 实时预测示例 ===")
    
    # 创建预测器
    config = {
        'input_dim': 4,
        'hidden_dim': 32,
        'output_dim': 3,
        'gcn_layers': 2,
        'lstm_layers': 1,
        'dropout': 0.1,
        'bidirectional': False,
        'fusion_weight': 0.6,
        'input_sequence_length': 6,
        'n_nodes': 15,
        'embedding_dim': 32,
        'history_length': 50,
        'bottleneck_threshold': 0.8,
        'propagation_speed_threshold': 10.0,
        'emergency_response_time': 300,
        'fundamental_diagram_type': 'triangular'
    }
    
    predictor = CongestionPropagationPredictor(config)
    
    # 模拟实时数据
    current_data = {}
    segments = create_sample_data(n_segments=15)
    
    for segment in segments:
        current_data[segment.segment_id] = {
            'length': segment.length,
            'lanes': segment.lanes,
            'capacity': segment.capacity,
            'free_flow_speed': segment.free_flow_speed,
            'current_speed': segment.current_speed,
            'current_flow': segment.current_flow,
            'occupancy': segment.occupancy
        }
    
    print("执行实时预测...")
    result = predictor.real_time_prediction(current_data, prediction_horizon=6)
    
    if result:
        print(f"实时预测结果 - 路段 {result.segment_id}:")
        print(f"  未来30分钟速度预测: {result.predicted_speeds}")
        print(f"  拥堵等级变化: {[level.name for level in result.congestion_levels]}")
        print(f"  平均置信度: {np.mean(result.confidence_scores):.3f}")
    else:
        print("实时预测失败")
    
    return predictor, result


def example_batch_prediction():
    """批量预测示例"""
    print("\n=== 批量预测示例 ===")
    
    # 创建预测器
    config = {
        'input_dim': 4,
        'hidden_dim': 48,
        'output_dim': 3,
        'gcn_layers': 2,
        'lstm_layers': 2,
        'dropout': 0.1,
        'bidirectional': True,
        'fusion_weight': 0.6,
        'input_sequence_length': 8,
        'n_nodes': 25,
        'embedding_dim': 48,
        'history_length': 80,
        'bottleneck_threshold': 0.75,
        'propagation_speed_threshold': 12.0,
        'emergency_response_time': 300,
        'fundamental_diagram_type': 'triangular'
    }
    
    predictor = CongestionPropagationPredictor(config)
    
    # 创建历史数据
    print("生成历史数据...")
    historical_data = []
    
    for time_point in range(5):  # 5个时间点
        segments = create_sample_data(n_segments=25)
        data_point = []
        
        for segment in segments:
            data_point.append({
                'segment_id': segment.segment_id,
                'length': segment.length,
                'lanes': segment.lanes,
                'capacity': segment.capacity,
                'free_flow_speed': segment.free_flow_speed,
                'current_speed': segment.current_speed,
                'current_flow': segment.current_flow,
                'occupancy': segment.occupancy
            })
        
        historical_data.append(data_point)
    
    print(f"生成了 {len(historical_data)} 个时间点的历史数据")
    
    # 执行批量预测
    print("执行批量预测...")
    results = predictor.batch_prediction(historical_data, prediction_horizon=4)
    
    print(f"批量预测完成，处理了 {len(historical_data)} 个时间点")
    print(f"生成 {len(results)} 个预测结果")
    
    # 分析批量结果
    time_point_results = {}
    for result in results:
        time_point = result.timestamp
        if time_point not in time_point_results:
            time_point_results[time_point] = []
        time_point_results[time_point].append(result)
    
    print(f"\n各时间点预测结果统计:")
    for time_point, time_results in time_point_results.items():
        avg_confidence = np.mean([r.confidence_scores.mean() for r in time_results])
        print(f"  时间点 {time_point}: {len(time_results)} 个路段, 平均置信度 {avg_confidence:.3f}")
    
    return predictor, results


def example_emergency_control():
    """应急控制策略示例"""
    print("\n=== 应急控制策略示例 ===")
    
    # 创建预测器
    config = {
        'input_dim': 4,
        'hidden_dim': 32,
        'output_dim': 3,
        'gcn_layers': 2,
        'lstm_layers': 1,
        'dropout': 0.1,
        'bidirectional': False,
        'fusion_weight': 0.6,
        'input_sequence_length': 6,
        'n_nodes': 20,
        'embedding_dim': 32,
        'history_length': 50,
        'bottleneck_threshold': 0.8,
        'propagation_speed_threshold': 10.0,
        'emergency_response_time': 300,
        'fundamental_diagram_type': 'triangular'
    }
    
    predictor = CongestionPropagationPredictor(config)
    
    # 创建包含瓶颈的路段
    segments = create_sample_data(n_segments=20)
    
    # 手动设置一些路段为瓶颈
    for i in [2, 5, 8, 12, 16]:
        if i < len(segments):
            segments[i].current_speed = segments[i].free_flow_speed * 0.3  # 严重拥堵
            segments[i].current_flow = segments[i].capacity * 1.2  # 超负荷
            segments[i].occupancy = 0.9
    
    # 识别瓶颈
    bottlenecks = predictor.identify_bottlenecks(segments)
    print(f"识别出 {len(bottlenecks)} 个瓶颈路段: {bottlenecks}")
    
    if not bottlenecks:
        print("未识别到瓶颈路段，手动选择几个路段进行应急策略演示")
        bottlenecks = ['segment_002', 'segment_005', 'segment_008']
    
    # 测试不同策略
    strategies = ['routing', 'signal', 'capacity']
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} 策略 ---")
        
        try:
            strategy_result = predictor.emergency_control_strategy(
                bottlenecks, segments, strategy
            )
            
            print(f"策略类型: {strategy_result['strategy_type']}")
            print(f"目标路段: {strategy_result['target_segments']}")
            print(f"预期效果: {strategy_result['effectiveness']:.1%}")
            print(f"实施时间: {strategy_result['implementation_time']} 秒")
            print(f"成本估算: {strategy_result['cost_estimate']:.0f} 元")
            
            if strategy == 'routing':
                print(f"交通分流: {strategy_result['traffic_diversion']}")
            elif strategy == 'signal':
                print(f"信号优化: {list(strategy_result['signal_optimization'].keys())}")
            elif strategy == 'capacity':
                print(f"能力提升: {list(strategy_result['capacity_enhancement'].keys())}")
                
        except Exception as e:
            print(f"策略 {strategy} 执行失败: {e}")
    
    return predictor, bottlenecks


def example_visualization():
    """可视化示例"""
    print("\n=== 可视化示例 ===")
    
    # 创建预测器
    config = {
        'input_dim': 4,
        'hidden_dim': 32,
        'output_dim': 3,
        'gcn_layers': 2,
        'lstm_layers': 1,
        'dropout': 0.1,
        'bidirectional': False,
        'fusion_weight': 0.6,
        'input_sequence_length': 6,
        'n_nodes': 15,
        'embedding_dim': 32,
        'history_length': 50,
        'bottleneck_threshold': 0.8,
        'propagation_speed_threshold': 10.0,
        'emergency_response_time': 300,
        'fundamental_diagram_type': 'triangular'
    }
    
    predictor = CongestionPropagationPredictor(config)
    
    # 创建示例数据
    segments = create_sample_data(n_segments=15)
    
    # 执行预测
    results = predictor.predict_congestion_propagation(
        segments, prediction_horizon=6, mode=PredictionMode.BATCH
    )
    
    # 可视化预测结果
    print("生成预测结果可视化...")
    visualize_prediction_results(results, '/workspace/code/models/example_prediction_results.png')
    
    # 创建自定义可视化
    create_custom_visualization(results, segments)
    
    return results


def create_custom_visualization(results, segments):
    """创建自定义可视化"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 提取数据
    segment_ids = [r.segment_id for r in results]
    speeds = np.array([r.predicted_speeds for r in results])
    flows = np.array([r.predicted_flows for r in results])
    occupancy = np.array([r.predicted_occupancy for r in results])
    confidence = np.array([r.confidence_scores for r in results])
    congestion_levels = np.array([[level.value for level in r.congestion_levels] for r in results])
    
    # 1. 速度热力图
    im1 = axes[0, 0].imshow(speeds.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    axes[0, 0].set_title('速度预测 (km/h)')
    axes[0, 0].set_xlabel('路段')
    axes[0, 0].set_ylabel('时间步')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 流量热力图
    im2 = axes[0, 1].imshow(flows.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title('流量预测 (veh/h)')
    axes[0, 1].set_xlabel('路段')
    axes[0, 1].set_ylabel('时间步')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. 占有率热力图
    im3 = axes[0, 2].imshow(occupancy.T, aspect='auto', cmap='plasma', interpolation='nearest')
    axes[0, 2].set_title('占有率预测')
    axes[0, 2].set_xlabel('路段')
    axes[0, 2].set_ylabel('时间步')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. 拥堵等级分布
    congestion_counts = np.sum(congestion_levels, axis=0)
    axes[1, 0].bar(range(len(congestion_counts)), congestion_counts)
    axes[1, 0].set_title('各时间步拥堵路段数量')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('拥堵路段数')
    
    # 5. 置信度分布
    axes[1, 1].hist(confidence.flatten(), bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('预测置信度分布')
    axes[1, 1].set_xlabel('置信度')
    axes[1, 1].set_ylabel('频次')
    
    # 6. 传播速度和影响范围散点图
    propagation_speeds = [r.propagation_speed for r in results]
    influence_ranges = [r.influence_range for r in results]
    
    scatter = axes[1, 2].scatter(propagation_speeds, influence_ranges, 
                                c=congestion_levels[:, -1], cmap='Reds', alpha=0.7)
    axes[1, 2].set_title('传播速度 vs 影响范围')
    axes[1, 2].set_xlabel('传播速度 (km/h)')
    axes[1, 2].set_ylabel('影响范围 (km)')
    plt.colorbar(scatter, ax=axes[1, 2], label='最终拥堵等级')
    
    plt.tight_layout()
    plt.savefig('/workspace/code/models/custom_visualization.png', dpi=300, bbox_inches='tight')
    print("自定义可视化已保存到: /workspace/code/models/custom_visualization.png")
    plt.show()


def example_model_persistence():
    """模型持久化示例"""
    print("\n=== 模型持久化示例 ===")
    
    # 创建预测器
    config = {
        'input_dim': 4,
        'hidden_dim': 32,
        'output_dim': 3,
        'gcn_layers': 2,
        'lstm_layers': 1,
        'dropout': 0.1,
        'bidirectional': False,
        'fusion_weight': 0.6,
        'input_sequence_length': 6,
        'n_nodes': 10,
        'embedding_dim': 32,
        'history_length': 50,
        'bottleneck_threshold': 0.8,
        'propagation_speed_threshold': 10.0,
        'emergency_response_time': 300,
        'fundamental_diagram_type': 'triangular'
    }
    
    predictor = CongestionPropagationPredictor(config)
    
    # 创建示例数据
    segments = create_sample_data(n_segments=10)
    
    # 执行一次预测以训练模型（实际应用中需要真实训练）
    print("执行初始预测...")
    results = predictor.predict_congestion_propagation(
        segments, prediction_horizon=3, mode=PredictionMode.BATCH
    )
    
    # 保存模型
    model_path = '/workspace/code/models/example_model.pth'
    predictor.save_model(model_path)
    
    # 创建新的预测器并加载模型
    print("创建新的预测器并加载模型...")
    new_predictor = CongestionPropagationPredictor(config)
    new_predictor.load_model(model_path)
    
    # 验证模型加载成功
    print("验证模型加载...")
    new_results = new_predictor.predict_congestion_propagation(
        segments, prediction_horizon=3, mode=PredictionMode.BATCH
    )
    
    print(f"原始预测结果数量: {len(results)}")
    print(f"加载模型后预测结果数量: {len(new_results)}")
    
    # 比较预测结果
    if results and new_results:
        speed_diff = np.mean(np.abs(results[0].predicted_speeds - new_results[0].predicted_speeds))
        print(f"速度预测差异: {speed_diff:.4f}")
    
    return predictor, new_predictor


def run_comprehensive_example():
    """运行综合示例"""
    print("=" * 60)
    print("拥堵扩散预测系统综合示例")
    print("=" * 60)
    
    try:
        # 1. 基本预测
        predictor1, results1, bottlenecks1 = example_basic_prediction()
        
        # 2. 实时预测
        predictor2, result2 = example_real_time_prediction()
        
        # 3. 批量预测
        predictor3, results3 = example_batch_prediction()
        
        # 4. 应急控制策略
        predictor4, bottlenecks4 = example_emergency_control()
        
        # 5. 可视化
        results5 = example_visualization()
        
        # 6. 模型持久化
        predictor6, new_predictor = example_model_persistence()
        
        print("\n" + "=" * 60)
        print("综合示例运行完成！")
        print("=" * 60)
        
        # 保存示例配置
        save_example_config()
        
    except Exception as e:
        print(f"示例运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def save_example_config():
    """保存示例配置"""
    example_config = {
        'model_config': {
            'input_dim': 4,
            'hidden_dim': 64,
            'output_dim': 3,
            'gcn_layers': 3,
            'lstm_layers': 2,
            'dropout': 0.1,
            'bidirectional': True,
            'fusion_weight': 0.6,
            'input_sequence_length': 12,
            'n_nodes': 30,
            'embedding_dim': 64,
            'history_length': 100,
            'bottleneck_threshold': 0.7,
            'propagation_speed_threshold': 15.0,
            'emergency_response_time': 300,
            'fundamental_diagram_type': 'triangular'
        },
        'prediction_settings': {
            'default_horizon': 6,
            'real_time_enabled': True,
            'batch_processing_enabled': True,
            'emergency_strategies': ['routing', 'signal', 'capacity'],
            'visualization_enabled': True
        },
        'performance_targets': {
            'max_prediction_time_seconds': 30,
            'min_accuracy_threshold': 0.75,
            'min_confidence_threshold': 0.6
        }
    }
    
    config_path = '/workspace/code/models/example_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)
    
    print(f"示例配置已保存到: {config_path}")


if __name__ == '__main__':
    run_comprehensive_example()