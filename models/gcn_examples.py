"""
GCN网络使用示例

该文件提供了GCN网络在实际应用中的使用示例，包括：
1. 基础使用示例
2. 交通流预测示例
3. 不同配置的对比示例
4. 模型保存和加载示例

Author: AI Assistant
Date: 2025-11-05
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import pickle

from gcn_network import (
    GraphDataProcessor,
    GCNNetwork,
    GCNTrainer,
    GCNEvaluator,
    create_sample_data
)


def basic_usage_example():
    """基础使用示例"""
    print("=== GCN网络基础使用示例 ===\n")
    
    # 1. 创建示例数据
    print("1. 创建示例数据...")
    data, coordinates = create_sample_data(
        n_timesteps=500, 
        n_nodes=30, 
        n_features=1
    )
    print(f"   数据形状: {data.shape}")
    print(f"   坐标形状: {coordinates.shape}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    processor = GraphDataProcessor(
        normalization='zscore',
        adj_threshold=0.1,
        sigma2=1.0,
        epsilon=0.1
    )
    
    graph_data = processor.prepare_graph_data(
        data, coordinates, 
        window_size=12, 
        prediction_steps=3
    )
    print(f"   输入数据: {graph_data['X'].shape}")
    print(f"   目标数据: {graph_data['y'].shape}")
    print(f"   邻接矩阵: {graph_data['adj_matrix'].shape}")
    
    # 3. 数据集划分
    print("\n3. 数据集划分...")
    X = graph_data['X']
    y = graph_data['y']
    
    n_samples = X.shape[0]
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X[:train_size])
    y_train = torch.FloatTensor(y[:train_size])
    X_val = torch.FloatTensor(X[train_size:train_size+val_size])
    y_val = torch.FloatTensor(y[train_size:train_size+val_size])
    X_test = torch.FloatTensor(X[train_size+val_size:])
    y_test = torch.FloatTensor(y[train_size+val_size:])
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")
    print(f"   测试集: {len(test_dataset)} 样本")
    
    # 4. 创建模型
    print("\n4. 创建GCN模型...")
    model = GCNNetwork(
        n_nodes=graph_data['n_nodes'],
        n_features=graph_data['n_features'],
        n_hidden=64,
        n_layers=3,
        conv_type='cheb',  # 可选: 'gcn', 'cheb', 'sage', 'gat'
        prediction_steps=graph_data['prediction_steps'],
        use_attention=True,
        use_dynamic_adj=True,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 5. 训练模型
    print("\n5. 训练模型...")
    trainer = GCNTrainer(model, learning_rate=0.001)
    
    # 准备图数据（转换为张量）
    graph_data_tensor = {
        'adj_matrix': torch.FloatTensor(graph_data['adj_matrix']),
        'laplacian': torch.FloatTensor(graph_data['laplacian']),
        'coordinates': torch.FloatTensor(graph_data['coordinates'])
    }
    
    # 训练模型
    training_history = trainer.train(
        train_loader, 
        val_loader, 
        graph_data_tensor,
        epochs=30,
        patience=10
    )
    
    # 6. 评估模型
    print("\n6. 评估模型...")
    evaluator = GCNEvaluator(model)
    results = evaluator.evaluate(test_loader, graph_data_tensor)
    
    # 打印评估结果
    print("   评估指标:")
    for metric, value in results['metrics'].items():
        print(f"     {metric}: {value:.6f}")
    
    # 7. 可视化结果
    print("\n7. 生成可视化结果...")
    evaluator.plot_results(results, save_path='basic_example_results.png')
    
    # 8. 保存模型
    print("\n8. 保存模型...")
    model_save_path = 'basic_gcn_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_nodes': graph_data['n_nodes'],
            'n_features': graph_data['n_features'],
            'n_hidden': 64,
            'n_layers': 3,
            'conv_type': 'cheb',
            'prediction_steps': graph_data['prediction_steps'],
            'use_attention': True,
            'use_dynamic_adj': True,
            'dropout': 0.1
        },
        'training_history': training_history,
        'metrics': results['metrics']
    }, model_save_path)
    print(f"   模型已保存到: {model_save_path}")
    
    return model, results, training_history


def traffic_prediction_example():
    """交通流预测示例"""
    print("\n=== 交通流预测示例 ===\n")
    
    # 模拟更真实的交通数据
    np.random.seed(42)
    
    # 1. 创建路网拓扑（模拟城市道路网络）
    n_nodes = 50
    coordinates = np.random.uniform(0, 1000, (n_nodes, 2))  # 城市区域 1km x 1km
    
    # 创建更真实的邻接矩阵（基于距离和连接性）
    def create_road_network(coordinates, connection_prob=0.1):
        n_nodes = len(coordinates)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # 基于距离的连接
        distances = np.linalg.norm(
            coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2
        )
        
        # 距离阈值
        distance_threshold = 200  # 200米
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if distances[i, j] < distance_threshold:
                    # 基于距离的权重
                    weight = np.exp(-distances[i, j] / 100)
                    if np.random.random() < connection_prob:
                        adj_matrix[i, j] = weight
                        adj_matrix[j, i] = weight
        
        return adj_matrix
    
    adj_matrix = create_road_network(coordinates)
    
    # 2. 生成交通流数据
    def generate_traffic_data(n_timesteps, n_nodes, adj_matrix):
        """生成模拟交通流数据"""
        data = np.zeros((n_timesteps, n_nodes, 3))  # 速度、流量、占用率
        
        # 模拟一天24小时 * 7天 = 168个时间点（每10分钟一个点）
        time_of_day = np.arange(n_timesteps) % 168
        
        for node in range(n_nodes):
            # 基于时间的基础模式
            base_speed = 60 + 20 * np.sin(2 * np.pi * time_of_day / 168)  # 速度变化
            base_flow = 1000 + 500 * np.sin(2 * np.pi * time_of_day / 168 + np.pi/4)  # 流量变化
            base_occupancy = 0.3 + 0.2 * np.sin(2 * np.pi * time_of_day / 168 + np.pi/2)  # 占用率变化
            
            # 添加邻接节点的影响
            neighbors = np.where(adj_matrix[node] > 0)[0]
            if len(neighbors) > 0:
                neighbor_effect = np.mean([base_speed[neighbor] for neighbor in neighbors], axis=0)
                base_speed = 0.7 * base_speed + 0.3 * neighbor_effect
            
            # 添加噪声
            speed_noise = np.random.normal(0, 5, n_timesteps)
            flow_noise = np.random.normal(0, 50, n_timesteps)
            occupancy_noise = np.random.normal(0, 0.05, n_timesteps)
            
            # 确保值在合理范围内
            speed = np.clip(base_speed + speed_noise, 10, 120)  # 10-120 km/h
            flow = np.clip(base_flow + flow_noise, 0, 2000)  # 0-2000 veh/h
            occupancy = np.clip(base_occupancy + occupancy_noise, 0.05, 0.95)  # 5%-95%
            
            data[:, node, 0] = speed
            data[:, node, 1] = flow
            data[:, node, 2] = occupancy
        
        return data
    
    # 生成一周的交通数据
    n_timesteps = 168 * 7  # 一周，每10分钟一个数据点
    traffic_data = generate_traffic_data(n_timesteps, n_nodes, adj_matrix)
    
    print(f"1. 生成交通数据: {traffic_data.shape}")
    print(f"   速度范围: {traffic_data[:,:,0].min():.1f} - {traffic_data[:,:,0].max():.1f} km/h")
    print(f"   流量范围: {traffic_data[:,:,1].min():.0f} - {traffic_data[:,:,1].max():.0f} veh/h")
    print(f"   占用率范围: {traffic_data[:,:,2].min():.2f} - {traffic_data[:,:,2].max():.2f}")
    
    # 3. 数据预处理
    print("\n2. 数据预处理...")
    processor = GraphDataProcessor(normalization='zscore')
    
    # 使用自定义邻接矩阵
    graph_data = processor.prepare_graph_data(traffic_data, coordinates, window_size=24, prediction_steps=6)
    
    # 替换为自定义的邻接矩阵
    graph_data['adj_matrix'] = adj_matrix
    graph_data['laplacian'] = processor._compute_laplacian(adj_matrix)
    
    print(f"   输入数据: {graph_data['X'].shape}")
    print(f"   预测步数: {graph_data['prediction_steps']} (相当于 {graph_data['prediction_steps'] * 10} 分钟)")
    
    # 4. 数据集划分（时间序列划分）
    X = graph_data['X']
    y = graph_data['y']
    
    n_samples = X.shape[0]
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    X_train = torch.FloatTensor(X[:train_size])
    y_train = torch.FloatTensor(y[:train_size])
    X_val = torch.FloatTensor(X[train_size:train_size+val_size])
    y_val = torch.FloatTensor(y[train_size:val_size+val_size])
    X_test = torch.FloatTensor(X[train_size+val_size:])
    y_test = torch.FloatTensor(y[train_size+val_size:])
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 5. 创建和训练模型
    print("\n3. 创建和训练模型...")
    model = GCNNetwork(
        n_nodes=graph_data['n_nodes'],
        n_features=graph_data['n_features'],  # 3: 速度、流量、占用率
        n_hidden=128,
        n_layers=4,
        conv_type='sage',  # GraphSAGE对交通数据效果较好
        prediction_steps=graph_data['prediction_steps'],
        use_attention=True,
        use_dynamic_adj=True,
        dropout=0.2
    )
    
    trainer = GCNTrainer(model, learning_rate=0.0005)
    
    graph_data_tensor = {
        'adj_matrix': torch.FloatTensor(graph_data['adj_matrix']),
        'laplacian': torch.FloatTensor(graph_data['laplacian']),
        'coordinates': torch.FloatTensor(graph_data['coordinates'])
    }
    
    # 训练模型
    training_history = trainer.train(
        train_loader, 
        val_loader, 
        graph_data_tensor,
        epochs=50,
        patience=15
    )
    
    # 6. 评估模型
    print("\n4. 评估模型...")
    evaluator = GCNEvaluator(model)
    results = evaluator.evaluate(test_loader, graph_data_tensor)
    
    # 分别评估不同特征的预测性能
    feature_names = ['速度', '流量', '占用率']
    print("   分特征评估结果:")
    for i, feature_name in enumerate(feature_names):
        pred_feature = results['predictions'][:, :, :, i]
        target_feature = results['targets'][:, :, :, i]
        
        mae = np.mean(np.abs(pred_feature - target_feature))
        rmse = np.sqrt(np.mean((pred_feature - target_feature) ** 2))
        
        print(f"     {feature_name}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
    
    # 7. 可视化路网和预测结果
    print("\n5. 生成可视化结果...")
    
    # 路网可视化
    plt.figure(figsize=(15, 5))
    
    # 路网拓扑
    plt.subplot(1, 3, 1)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=30, alpha=0.7)
    
    # 绘制边
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adj_matrix[i, j] > 0:
                plt.plot([coordinates[i, 0], coordinates[j, 0]], 
                        [coordinates[i, 1], coordinates[j, 1]], 
                        'b-', alpha=0.3, linewidth=0.5)
    
    plt.title('路网拓扑')
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    
    # 预测结果示例
    plt.subplot(1, 3, 2)
    sample_idx = 0
    node_idx = 0
    time_steps = range(graph_data['prediction_steps'])
    
    pred_speed = results['predictions'][sample_idx, :, node_idx, 0]
    target_speed = results['targets'][sample_idx, :, node_idx, 0]
    
    plt.plot(time_steps, target_speed, 'b-', label='实际速度', linewidth=2)
    plt.plot(time_steps, pred_speed, 'r--', label='预测速度', linewidth=2)
    plt.title(f'速度预测示例 (节点 {node_idx})')
    plt.xlabel('预测步数')
    plt.ylabel('速度 (km/h)')
    plt.legend()
    
    # 性能指标
    plt.subplot(1, 3, 3)
    metrics = results['metrics']
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = plt.bar(metric_names, metric_values)
    plt.title('模型性能指标')
    plt.ylabel('值')
    plt.xticks(rotation=45)
    
    # 在条形图上显示数值
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('traffic_prediction_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, results, graph_data


def model_comparison_example():
    """模型配置对比示例"""
    print("\n=== 模型配置对比示例 ===\n")
    
    # 创建示例数据
    data, coordinates = create_sample_data(n_timesteps=300, n_nodes=20, n_features=1)
    
    # 数据预处理
    processor = GraphDataProcessor(normalization='zscore')
    graph_data = processor.prepare_graph_data(data, coordinates, window_size=10, prediction_steps=2)
    
    # 数据集划分
    X = graph_data['X']
    y = graph_data['y']
    
    n_samples = X.shape[0]
    train_size = int(0.7 * n_samples)
    
    X_train = torch.FloatTensor(X[:train_size])
    y_train = torch.FloatTensor(y[:train_size])
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    graph_data_tensor = {
        'adj_matrix': torch.FloatTensor(graph_data['adj_matrix']),
        'laplacian': torch.FloatTensor(graph_data['laplacian']),
        'coordinates': torch.FloatTensor(graph_data['coordinates'])
    }
    
    # 配置对比
    configs = [
        {
            'name': 'GCN + LSTM',
            'conv_type': 'gcn',
            'use_attention': False,
            'use_dynamic_adj': False,
            'n_hidden': 32,
            'n_layers': 2
        },
        {
            'name': 'ChebNet + Attention',
            'conv_type': 'cheb',
            'use_attention': True,
            'use_dynamic_adj': False,
            'n_hidden': 32,
            'n_layers': 2
        },
        {
            'name': 'GraphSAGE + Dynamic',
            'conv_type': 'sage',
            'use_attention': False,
            'use_dynamic_adj': True,
            'n_hidden': 32,
            'n_layers': 2
        },
        {
            'name': 'GAT + Full Features',
            'conv_type': 'gat',
            'use_attention': True,
            'use_dynamic_adj': True,
            'n_hidden': 32,
            'n_layers': 2
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"训练配置: {config['name']}")
        
        # 创建模型
        model = GCNNetwork(
            n_nodes=graph_data['n_nodes'],
            n_features=graph_data['n_features'],
            n_hidden=config['n_hidden'],
            n_layers=config['n_layers'],
            conv_type=config['conv_type'],
            prediction_steps=graph_data['prediction_steps'],
            use_attention=config['use_attention'],
            use_dynamic_adj=config['use_dynamic_adj'],
            dropout=0.1
        )
        
        # 创建训练器
        trainer = GCNTrainer(model, learning_rate=0.001)
        
        # 训练
        train_losses = []
        for epoch in range(20):
            train_loss = trainer.train_epoch(train_loader, graph_data_tensor)
            train_losses.append(train_loss)
        
        # 记录结果
        results.append({
            'config': config,
            'final_loss': train_losses[-1],
            'train_losses': train_losses,
            'param_count': sum(p.numel() for p in model.parameters())
        })
        
        print(f"  最终训练损失: {train_losses[-1]:.6f}")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters())}")
        print()
    
    # 可视化对比结果
    plt.figure(figsize=(12, 8))
    
    # 训练损失曲线
    plt.subplot(2, 2, 1)
    for result in results:
        plt.plot(result['train_losses'], label=result['config']['name'])
    plt.title('训练损失曲线对比')
    plt.xlabel('Epoch')
    plt.ylabel('训练损失')
    plt.legend()
    plt.yscale('log')
    
    # 最终损失对比
    plt.subplot(2, 2, 2)
    config_names = [r['config']['name'] for r in results]
    final_losses = [r['final_loss'] for r in results]
    bars = plt.bar(config_names, final_losses)
    plt.title('最终训练损失对比')
    plt.ylabel('训练损失')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, final_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    # 参数数量对比
    plt.subplot(2, 2, 3)
    param_counts = [r['param_count'] for r in results]
    bars = plt.bar(config_names, param_counts)
    plt.title('模型参数数量对比')
    plt.ylabel('参数数量')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, param_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}', ha='center', va='bottom')
    
    # 性能总结表
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 创建总结表格
    table_data = []
    for result in results:
        config = result['config']
        table_data.append([
            config['name'],
            f"{result['final_loss']:.4f}",
            f"{result['param_count']:,}",
            config['conv_type'].upper(),
            "✓" if config['use_attention'] else "✗",
            "✓" if config['use_dynamic_adj'] else "✗"
        ])
    
    headers = ['配置', '最终损失', '参数数量', '卷积类型', '注意力', '动态邻接']
    
    table = plt.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title('配置对比总结')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印总结
    print("=== 配置对比总结 ===")
    print(f"{'配置':<20} {'最终损失':<12} {'参数数量':<10} {'卷积类型':<12} {'注意力':<8} {'动态邻接':<8}")
    print("-" * 80)
    for result in results:
        config = result['config']
        print(f"{config['name']:<20} {result['final_loss']:<12.6f} {result['param_count']:<10,} "
              f"{config['conv_type'].upper():<12} {'✓' if config['use_attention'] else '✗':<8} "
              f"{'✓' if config['use_dynamic_adj'] else '✗':<8}")
    
    return results


def save_and_load_example():
    """模型保存和加载示例"""
    print("\n=== 模型保存和加载示例 ===\n")
    
    # 1. 训练一个模型
    print("1. 训练基础模型...")
    data, coordinates = create_sample_data(n_timesteps=200, n_nodes=15, n_features=1)
    
    processor = GraphDataProcessor(normalization='zscore')
    graph_data = processor.prepare_graph_data(data, coordinates, window_size=8, prediction_steps=1)
    
    # 快速训练
    X = torch.FloatTensor(graph_data['X'][:50])
    y = torch.FloatTensor(graph_data['y'][:50])
    
    train_dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    model = GCNNetwork(
        n_nodes=graph_data['n_nodes'],
        n_features=graph_data['n_features'],
        n_hidden=32,
        n_layers=2,
        conv_type='cheb',
        prediction_steps=graph_data['prediction_steps'],
        use_attention=True,
        use_dynamic_adj=False,
        dropout=0.1
    )
    
    trainer = GCNTrainer(model, learning_rate=0.001)
    
    graph_data_tensor = {
        'adj_matrix': torch.FloatTensor(graph_data['adj_matrix']),
        'laplacian': torch.FloatTensor(graph_data['laplacian']),
        'coordinates': torch.FloatTensor(graph_data['coordinates'])
    }
    
    # 训练几个epoch
    for epoch in range(10):
        trainer.train_epoch(train_loader, graph_data_tensor)
    
    print("   模型训练完成")
    
    # 2. 保存模型
    print("\n2. 保存模型...")
    
    # 保存完整模型信息
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_nodes': graph_data['n_nodes'],
            'n_features': graph_data['n_features'],
            'n_hidden': 32,
            'n_layers': 2,
            'conv_type': 'cheb',
            'prediction_steps': graph_data['prediction_steps'],
            'use_attention': True,
            'use_dynamic_adj': False,
            'dropout': 0.1
        },
        'preprocessing_params': {
            'normalization': 'zscore',
            'adj_threshold': 0.1,
            'sigma2': 1.0,
            'epsilon': 0.1
        },
        'training_info': {
            'epochs_trained': 10,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        }
    }
    
    # 保存为不同格式
    torch.save(model_info, 'gcn_model_complete.pth')
    
    # 只保存模型参数
    torch.save(model.state_dict(), 'gcn_model_weights.pth')
    
    # 保存模型配置
    config_path = 'gcn_model_config.json'
    with open(config_path, 'w') as f:
        json.dump(model_info['model_config'], f, indent=2)
    
    print(f"   完整模型信息保存到: gcn_model_complete.pth")
    print(f"   模型权重保存到: gcn_model_weights.pth")
    print(f"   模型配置保存到: {config_path}")
    
    # 3. 加载模型
    print("\n3. 加载模型...")
    
    # 加载完整模型信息
    loaded_info = torch.load('gcn_model_complete.pth')
    
    # 创建新模型
    new_model = GCNNetwork(**loaded_info['model_config'])
    new_model.load_state_dict(loaded_info['model_state_dict'])
    
    print("   完整模型信息加载成功")
    
    # 验证模型是否加载正确
    new_model.eval()
    with torch.no_grad():
        test_input = X[:2]
        original_output = model(test_input, graph_data_tensor)
        loaded_output = new_model(test_input, graph_data_tensor)
        
        # 检查输出是否一致
        diff = torch.abs(original_output['predictions'] - loaded_output['predictions']).max()
        print(f"   模型验证: 最大输出差异 = {diff:.8f}")
        
        if diff < 1e-6:
            print("   ✓ 模型加载验证通过")
        else:
            print("   ✗ 模型加载验证失败")
    
    # 4. 模型推理示例
    print("\n4. 模型推理示例...")
    
    # 准备新的测试数据
    new_data, new_coordinates = create_sample_data(n_timesteps=50, n_nodes=15, n_features=1)
    new_graph_data = processor.prepare_graph_data(new_data, new_coordinates, window_size=8, prediction_steps=1)
    
    new_graph_data_tensor = {
        'adj_matrix': torch.FloatTensor(new_graph_data['adj_matrix']),
        'laplacian': torch.FloatTensor(new_graph_data['laplacian']),
        'coordinates': torch.FloatTensor(new_graph_data['coordinates'])
    }
    
    test_input = torch.FloatTensor(new_graph_data['X'][:5])
    
    # 使用加载的模型进行推理
    new_model.eval()
    with torch.no_grad():
        predictions = new_model(test_input, new_graph_data_tensor)
    
    print(f"   推理输入形状: {test_input.shape}")
    print(f"   预测输出形状: {predictions['predictions'].shape}")
    print(f"   预测值范围: [{predictions['predictions'].min():.4f}, {predictions['predictions'].max():.4f}]")
    
    # 5. 模型信息总结
    print("\n5. 模型信息总结...")
    print(f"   模型类型: GCN Network")
    print(f"   卷积类型: {loaded_info['model_config']['conv_type']}")
    print(f"   隐藏维度: {loaded_info['model_config']['n_hidden']}")
    print(f"   层数: {loaded_info['model_config']['n_layers']}")
    print(f"   注意力: {'启用' if loaded_info['model_config']['use_attention'] else '禁用'}")
    print(f"   动态邻接: {'启用' if loaded_info['model_config']['use_dynamic_adj'] else '禁用'}")
    print(f"   训练轮数: {loaded_info['training_info']['epochs_trained']}")
    print(f"   学习率: {loaded_info['training_info']['learning_rate']}")
    
    return new_model, predictions


def main():
    """主函数 - 运行所有示例"""
    print("=" * 60)
    print("GCN网络使用示例集合")
    print("=" * 60)
    
    try:
        # 1. 基础使用示例
        basic_model, basic_results, basic_history = basic_usage_example()
        
        # 2. 交通流预测示例
        traffic_model, traffic_results, traffic_graph_data = traffic_prediction_example()
        
        # 3. 模型配置对比示例
        comparison_results = model_comparison_example()
        
        # 4. 模型保存和加载示例
        loaded_model, inference_results = save_and_load_example()
        
        print("\n" + "=" * 60)
        print("🎉 所有示例运行完成！")
        print("=" * 60)
        print("\n生成的文件:")
        print("- basic_example_results.png: 基础示例结果")
        print("- traffic_prediction_example.png: 交通预测示例")
        print("- model_comparison.png: 模型对比结果")
        print("- gcn_model_*.pth: 保存的模型文件")
        print("- gcn_model_config.json: 模型配置文件")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()