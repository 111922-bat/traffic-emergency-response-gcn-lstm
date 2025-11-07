#!/usr/bin/env python3
"""详细调试测试"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../models'))

from gcn_lstm_hybrid import (
    GCNLSTMHybrid, ModelConfig, FusionStrategy, TaskType,
    create_sample_adj_matrix, GraphConvLayer
)

def detailed_debug():
    print("=== 详细调试测试 ===")
    
    # 创建配置
    config = ModelConfig(
        fusion_strategy=FusionStrategy.ATTENTION,
        task_types=[TaskType.SPEED_PREDICTION, TaskType.CONGESTION_PREDICTION],
        use_dynamic_graph=True
    )
    
    # 创建测试数据
    batch_size, seq_len, num_nodes = 2, 8, 30
    input_data = torch.randn(batch_size, seq_len, num_nodes, config.input_dim)
    adj_matrix = create_sample_adj_matrix(num_nodes)
    
    print(f"输入数据形状: {input_data.shape}")
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    
    # 测试单个GraphConvLayer
    print("\n1. 测试单个GraphConvLayer...")
    try:
        gcn_layer = GraphConvLayer(
            input_dim=config.input_dim,
            output_dim=config.hidden_dim,
            dropout=config.gcn_dropout
        )
        
        print(f"   输入数据形状: {input_data.shape}")
        print(f"   邻接矩阵形状: {adj_matrix.shape}")
        
        # 手动执行forward逻辑
        batch_size, seq_len, num_nodes, _ = input_data.shape
        print(f"   批次大小: {batch_size}, 序列长度: {seq_len}, 节点数: {num_nodes}")
        
        # 重塑输入
        x_reshaped = input_data.view(batch_size * seq_len, num_nodes, config.input_dim)
        print(f"   重塑后输入形状: {x_reshaped.shape}")
        
        # 计算图卷积
        support = torch.matmul(x_reshaped, gcn_layer.weight)
        print(f"   支持矩阵形状: {support.shape}")
        
        # 扩展邻接矩阵
        adj_expanded = adj_matrix.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        print(f"   扩展后邻接矩阵形状: {adj_expanded.shape}")
        
        # 执行图卷积
        output = torch.matmul(adj_expanded, support)
        print(f"   图卷积结果形状: {output.shape}")
        
        # 添加偏置
        output = output + gcn_layer.bias
        print(f"   添加偏置后形状: {output.shape}")
        
        # 重塑回原始形状
        output = output.view(batch_size, seq_len, num_nodes, config.hidden_dim)
        print(f"   最终输出形状: {output.shape}")
        
        print("   ✓ 单个GraphConvLayer测试成功")
        
    except Exception as e:
        print(f"   ✗ 单个GraphConvLayer失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试完整的GCNModule
    print("\n2. 测试完整GCNModule...")
    try:
        from gcn_lstm_hybrid import GCNModule
        gcn_module = GCNModule(config)
        gcn_output = gcn_module(input_data, adj_matrix)
        print(f"   ✓ GCN模块输出形状: {gcn_output.shape}")
    except Exception as e:
        print(f"   ✗ GCN模块失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== 详细调试完成 ===")
    return True

if __name__ == "__main__":
    detailed_debug()