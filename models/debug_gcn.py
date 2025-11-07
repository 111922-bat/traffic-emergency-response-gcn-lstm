"""
简化的GCN测试，用于调试维度问题
"""

import torch
import numpy as np
from gcn_network import ChebConv

def debug_chebconv():
    """调试ChebConv的维度问题"""
    print("=== 调试ChebConv维度问题 ===")
    
    # 创建测试数据
    batch_size, n_nodes, in_features, out_features = 4, 20, 16, 32
    
    # 创建测试数据
    x = torch.randn(batch_size, n_nodes, in_features)
    print(f"x形状: {x.shape}")
    print(f"x的实际数据指针: {x.data_ptr()}")
    
    # 创建邻接矩阵和拉普拉斯矩阵
    np.random.seed(42)
    adj_matrix = np.random.uniform(0, 1, (n_nodes, n_nodes))
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 对称化
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix = torch.FloatTensor(adj_matrix)
    
    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
    laplacian = degree_matrix - adj_matrix
    
    print(f"adj_matrix形状: {adj_matrix.shape}")
    print(f"laplacian形状: {laplacian.shape}")
    
    # 创建ChebConv
    cheb_conv = ChebConv(in_features, out_features, k=3)
    print(f"weight形状: {cheb_conv.weight.shape}")
    
    # 打印权重形状的详细信息
    for i, w in enumerate(cheb_conv.weight):
        print(f"weight[{i}]形状: {w.shape}")
    
    # 手动实现前向传播来调试
    print("\n=== 手动实现前向传播 ===")
    
    batch_size, n_nodes, in_features = x.shape
    print(f"从x中提取的维度: batch_size={batch_size}, n_nodes={n_nodes}, in_features={in_features}")
    
    # 确保laplacian矩阵维度正确
    if laplacian.shape[0] != n_nodes:
        laplacian = laplacian[:n_nodes, :n_nodes]
        print(f"调整后的laplacian形状: {laplacian.shape}")
    
    # 归一化拉普拉斯矩阵
    lambda_max = 2.0
    laplacian_normalized = laplacian / lambda_max
    print(f"laplacian_normalized形状: {laplacian_normalized.shape}")
    
    # 使用一阶近似
    first_order = torch.eye(n_nodes).to(laplacian.device) + laplacian_normalized
    print(f"first_order形状: {first_order.shape}")
    
    # 确保矩阵维度匹配
    if first_order.shape != (n_nodes, n_nodes):
        first_order = first_order[:n_nodes, :n_nodes]
        print(f"调整后的first_order形状: {first_order.shape}")
    
    # 尝试矩阵乘法
    print(f"\n尝试矩阵乘法:")
    print(f"x.shape: {x.shape}")
    print(f"first_order.shape: {first_order.shape}")
    
    try:
        # 先尝试简单的矩阵乘法
        print("尝试1: torch.matmul(x, first_order)")
        result1 = torch.matmul(x, first_order)
        print(f"成功! result1.shape: {result1.shape}")
        
        # 然后与权重相乘
        print("尝试2: torch.matmul(result1, cheb_conv.weight[0])")
        result2 = torch.matmul(result1, cheb_conv.weight[0])
        print(f"成功! result2.shape: {result2.shape}")
        
        print("手动计算成功!")
        
    except Exception as e:
        print(f"手动计算失败: {e}")
        
        # 尝试其他方法
        print("\n尝试其他方法:")
        
        # 尝试reshape x
        x_reshaped = x.view(-1, in_features)  # [batch*n_nodes, in_features]
        print(f"x_reshaped.shape: {x_reshaped.shape}")
        
        try:
            result3 = torch.matmul(x_reshaped, first_order)
            print(f"reshape后成功! result3.shape: {result3.shape}")
            
            # 再reshape回来
            result4 = result3.view(batch_size, n_nodes, n_nodes)
            print(f"reshape back成功! result4.shape: {result4.shape}")
            
        except Exception as e2:
            print(f"reshape方法也失败: {e2}")

if __name__ == "__main__":
    debug_chebconv()