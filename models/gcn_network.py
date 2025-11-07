"""
GCN图卷积网络模块实现

基于研究结果实现的完整GCN网络，支持多种图卷积操作：
- ChebNet（切比雪夫图卷积）
- GraphSAGE（图采样聚合）
- GAT（图注意力网络）
- 动态邻接矩阵学习
- 多头注意力机制
- 图结构数据预处理和标准化

Author: AI Assistant
Date: 2025-11-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class GraphDataProcessor:
    """图数据预处理和标准化类"""
    
    def __init__(self, 
                 normalization: str = 'zscore',
                 adj_threshold: float = 0.1,
                 sigma2: float = 1.0,
                 epsilon: float = 0.1):
        """
        初始化图数据处理器
        
        Args:
            normalization: 归一化方法 ('zscore', 'minmax', 'none')
            adj_threshold: 邻接矩阵阈值
            sigma2: 高斯核参数
            epsilon: 距离阈值参数
        """
        self.normalization = normalization
        self.adj_threshold = adj_threshold
        self.sigma2 = sigma2
        self.epsilon = epsilon
        self.scaler = None
        
    def build_adjacency_matrix(self, 
                              coordinates: np.ndarray,
                              method: str = 'distance') -> np.ndarray:
        """
        构建邻接矩阵
        
        Args:
            coordinates: 节点坐标 [n_nodes, 2]
            method: 构建方法 ('distance', 'connectivity', 'adaptive')
            
        Returns:
            adj_matrix: 邻接矩阵 [n_nodes, n_nodes]
        """
        n_nodes = coordinates.shape[0]
        
        if method == 'distance':
            # 基于距离的高斯核构建
            distances = squareform(pdist(coordinates, metric='euclidean'))
            adj_matrix = np.exp(-distances**2 / (2 * self.sigma2))
            adj_matrix[adj_matrix < self.epsilon] = 0
            np.fill_diagonal(adj_matrix, 1)
            
        elif method == 'connectivity':
            # 基于k近邻的连接性
            distances = squareform(pdist(coordinates, metric='euclidean'))
            k = min(10, n_nodes - 1)  # k近邻
            adj_matrix = np.zeros((n_nodes, n_nodes))
            
            for i in range(n_nodes):
                k_nearest = np.argsort(distances[i])[:k+1]
                k_nearest = k_nearest[k_nearest != i]  # 排除自己
                adj_matrix[i, k_nearest] = 1
                adj_matrix[k_nearest, i] = 1
                
        elif method == 'adaptive':
            # 自适应邻接矩阵（端到端学习）
            # 这里返回初始化的自适应矩阵，实际学习中会更新
            adj_matrix = np.random.uniform(0, 1, (n_nodes, n_nodes))
            adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 对称化
            np.fill_diagonal(adj_matrix, 0)
            
        else:
            raise ValueError(f"Unknown adjacency method: {method}")
            
        return adj_matrix
    
    def normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        数据归一化
        
        Args:
            data: 输入数据 [n_samples, n_nodes, n_features]
            fit: 是否拟合归一化器
            
        Returns:
            normalized_data: 归一化后的数据
        """
        n_samples, n_nodes, n_features = data.shape
        
        # 重塑为2D进行归一化
        data_2d = data.reshape(-1, n_features)
        
        if fit:
            if self.normalization == 'zscore':
                self.scaler = StandardScaler()
            elif self.normalization == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = None
                
            if self.scaler is not None:
                data_2d = self.scaler.fit_transform(data_2d)
                
        else:
            if self.scaler is not None:
                data_2d = self.scaler.transform(data_2d)
                
        # 重塑回原始形状
        normalized_data = data_2d.reshape(n_samples, n_nodes, n_features)
        return normalized_data
    
    def create_sliding_window(self, 
                             data: np.ndarray, 
                             window_size: int = 12, 
                             prediction_steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口数据
        
        Args:
            data: 输入时序数据 [n_timesteps, n_nodes, n_features]
            window_size: 历史窗口大小
            prediction_steps: 预测步数
            
        Returns:
            X: 输入序列 [n_samples, window_size, n_nodes, n_features]
            y: 目标序列 [n_samples, prediction_steps, n_nodes, n_features]
        """
        n_timesteps, n_nodes, n_features = data.shape
        n_samples = n_timesteps - window_size - prediction_steps + 1
        
        X = np.zeros((n_samples, window_size, n_nodes, n_features))
        y = np.zeros((n_samples, prediction_steps, n_nodes, n_features))
        
        for i in range(n_samples):
            X[i] = data[i:i+window_size]
            y[i] = data[i+window_size:i+window_size+prediction_steps]
            
        return X, y
    
    def prepare_graph_data(self, 
                          data: np.ndarray,
                          coordinates: np.ndarray,
                          window_size: int = 12,
                          prediction_steps: int = 1) -> Dict:
        """
        准备图数据
        
        Args:
            data: 时序数据 [n_timesteps, n_nodes, n_features]
            coordinates: 节点坐标 [n_nodes, 2]
            window_size: 历史窗口大小
            prediction_steps: 预测步数
            
        Returns:
            graph_data: 包含所有图数据的字典
        """
        # 构建邻接矩阵
        adj_matrix = self.build_adjacency_matrix(coordinates)
        
        # 数据归一化
        normalized_data = self.normalize_data(data, fit=True)
        
        # 创建滑动窗口
        X, y = self.create_sliding_window(normalized_data, window_size, prediction_steps)
        
        # 计算图的拉普拉斯矩阵
        laplacian = self._compute_laplacian(adj_matrix)
        
        graph_data = {
            'X': X,  # [n_samples, window_size, n_nodes, n_features]
            'y': y,  # [n_samples, prediction_steps, n_nodes, n_features]
            'adj_matrix': adj_matrix,  # [n_nodes, n_nodes]
            'laplacian': laplacian,  # [n_nodes, n_nodes]
            'coordinates': coordinates,  # [n_nodes, 2]
            'n_nodes': coordinates.shape[0],
            'n_features': data.shape[2],
            'window_size': window_size,
            'prediction_steps': prediction_steps
        }
        
        return graph_data
    
    def _compute_laplacian(self, adj_matrix: np.ndarray) -> np.ndarray:
        """计算图的拉普拉斯矩阵"""
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix
        return laplacian


class ChebConv(nn.Module):
    """切比雪夫图卷积层"""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 k: int = 3,
                 bias: bool = True):
        """
        初始化切比雪夫卷积层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            k: 切比雪夫多项式阶数
            bias: 是否使用偏置
        """
        super(ChebConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        
        self.weight = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, n_nodes, in_features]
            laplacian: 拉普拉斯矩阵 [n_nodes, n_nodes]
            
        Returns:
            out: 输出特征 [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, in_features = x.shape
        
        # 确保laplacian矩阵维度正确
        if laplacian.shape[0] != n_nodes:
            # 如果维度不匹配，使用子矩阵
            laplacian = laplacian[:n_nodes, :n_nodes]
        
        # 归一化拉普拉斯矩阵
        lambda_max = 2.0
        laplacian_normalized = laplacian / lambda_max
        
        # 简化的图卷积实现
        out = torch.zeros(batch_size, n_nodes, self.out_features).to(x.device)
        
        # 使用一阶近似: (I + L) * x
        first_order = torch.eye(n_nodes).to(laplacian.device) + laplacian_normalized
        
        # 确保矩阵维度匹配
        if first_order.shape != (n_nodes, n_nodes):
            first_order = first_order[:n_nodes, :n_nodes]
        
        # 图卷积: adjacency @ node_features
        # first_order: [n_nodes, n_nodes] 
        # x: [batch, n_nodes, in_features]
        # 对于每个batch和特征维度应用图卷积
        x_permuted = x.transpose(1, 2)  # [batch, in_features, n_nodes]
        # 矩阵乘法: [n_nodes, n_nodes] @ [batch, in_features, n_nodes] -> [batch, in_features, n_nodes]
        first_order_x = torch.matmul(first_order, x_permuted)  # [batch, in_features, n_nodes]
        first_order_x = first_order_x.transpose(1, 2)  # [batch, n_nodes, in_features]
        
        # 特征变换: [batch, n_nodes, in_features] @ [in_features, out_features] -> [batch, n_nodes, out_features]
        out = out + torch.matmul(first_order_x, self.weight[0])
        
        # 如果有多阶权重，使用二阶近似
        if self.k > 1 and self.weight.shape[0] > 1:
            second_order = 2 * torch.mm(laplacian_normalized, first_order) - torch.eye(n_nodes).to(laplacian.device)
            if second_order.shape != (n_nodes, n_nodes):
                second_order = second_order[:n_nodes, :n_nodes]
            second_order_x = torch.matmul(second_order, x_permuted).transpose(1, 2)  # [batch, n_nodes, in_features]
            out = out + torch.matmul(second_order_x, self.weight[1])  # [batch, n_nodes, out_features]
        
        if self.bias is not None:
            out = out + self.bias
            
        return out


class GraphSAGEConv(nn.Module):
    """GraphSAGE图卷积层"""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 aggregator: str = 'mean',
                 dropout: float = 0.0):
        """
        初始化GraphSAGE卷积层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            aggregator: 聚合方法 ('mean', 'max', 'sum')
            dropout: dropout概率
        """
        super(GraphSAGEConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, n_nodes, in_features]
            adj_matrix: 邻接矩阵 [n_nodes, n_nodes]
            
        Returns:
            out: 输出特征 [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, in_features = x.shape
        
        # 自身特征变换
        self_out = self.linear_self(x)
        
        # 邻居特征聚合
        neigh_out = torch.zeros_like(self_out)
        
        for i in range(n_nodes):
            neighbors = torch.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                neighbor_features = x[:, neighbors, :]  # [batch, n_neighbors, in_features]
                
                if self.aggregator == 'mean':
                    agg_features = torch.mean(neighbor_features, dim=1)
                elif self.aggregator == 'max':
                    agg_features = torch.max(neighbor_features, dim=1)[0]
                elif self.aggregator == 'sum':
                    agg_features = torch.sum(neighbor_features, dim=1)
                else:
                    raise ValueError(f"Unknown aggregator: {self.aggregator}")
                
                neigh_out[:, i, :] = self.linear_neigh(agg_features)
        
        # 合并自身和邻居特征
        out = self_out + neigh_out
        out = F.relu(out)
        out = self.dropout(out)
        
        return out


class GATConv(nn.Module):
    """图注意力网络卷积层"""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 leaky_relu: float = 0.2):
        """
        初始化图注意力卷积层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            n_heads: 注意力头数
            dropout: dropout概率
            leaky_relu: LeakyReLU负斜率
        """
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, n_nodes, in_features]
            adj_matrix: 邻接矩阵 [n_nodes, n_nodes]
            
        Returns:
            out: 输出特征 [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, in_features = x.shape
        
        # 准备注意力掩码
        mask = (adj_matrix == 0).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 应用注意力机制
        attended_x, _ = self.attention(x, x, x, attn_mask=mask)
        attended_x = self.dropout(attended_x)
        
        # 特征变换
        out = self.linear(attended_x)
        out = self.leaky_relu(out)
        
        return out


class DynamicAdjacencyLearner(nn.Module):
    """动态邻接矩阵学习模块"""
    
    def __init__(self, 
                 n_nodes: int,
                 hidden_dim: int = 64,
                 n_heads: int = 8):
        """
        初始化动态邻接矩阵学习器
        
        Args:
            n_nodes: 节点数
            hidden_dim: 隐藏层维度
            n_heads: 注意力头数
        """
        super(DynamicAdjacencyLearner, self).__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # 节点嵌入
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, hidden_dim))
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            for _ in range(2)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        学习动态邻接矩阵
        
        Args:
            x: 输入特征 [batch, n_nodes, features]
            
        Returns:
            adj_matrix: 动态邻接矩阵 [batch, n_nodes, n_nodes]
        """
        batch_size, n_nodes, features = x.shape
        
        # 扩展节点嵌入
        node_emb = self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 第一层注意力
        h1, _ = self.attention_layers[0](node_emb, node_emb, node_emb)
        
        # 第二层注意力
        h2, _ = self.attention_layers[1](h1, h1, h1)
        
        # 计算注意力权重
        scores = torch.einsum('bij,bkj->bik', h2, h2)  # [batch, n_nodes, n_nodes]
        
        # 应用输出层
        scores = self.output_layer(scores.unsqueeze(-1)).squeeze(-1)
        
        # 转换为邻接矩阵（对称化并确保非负）
        adj_matrix = torch.sigmoid(scores)
        adj_matrix = (adj_matrix + adj_matrix.transpose(1, 2)) / 2  # 对称化
        
        # 对角线设为0（避免自环）
        mask = torch.eye(n_nodes).unsqueeze(0).expand(batch_size, -1, -1).to(adj_matrix.device)
        adj_matrix = adj_matrix * (1 - mask)
        
        return adj_matrix


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 output_dim: int = None):
        """
        初始化多头注意力层
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: dropout概率
            output_dim: 输出维度（可选）
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim or embed_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(embed_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, 
                key: torch.Tensor = None, 
                value: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch, seq_len, embed_dim]
            key: 键张量 [batch, seq_len, embed_dim]
            value: 值张量 [batch, seq_len, embed_dim]
            mask: 注意力掩码
            
        Returns:
            output: 输出张量 [batch, seq_len, output_dim]
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        # 应用多头注意力
        attended, attention_weights = self.attention(query, key, value, attn_mask=mask)
        
        # 输出投影
        output = self.output_projection(attended)
        output = self.dropout(output)
        
        return output, attention_weights


class GCNBlock(nn.Module):
    """GCN基础块"""
    
    def __init__(self, 
                 conv_type: str,
                 in_features: int,
                 out_features: int,
                 **kwargs):
        """
        初始化GCN块
        
        Args:
            conv_type: 卷积类型 ('cheb', 'sage', 'gat', 'gcn')
            in_features: 输入特征维度
            out_features: 输出特征维度
            **kwargs: 其他参数
        """
        super(GCNBlock, self).__init__()
        self.conv_type = conv_type
        
        if conv_type == 'cheb':
            self.conv = ChebConv(in_features, out_features, **kwargs)
        elif conv_type == 'sage':
            self.conv = GraphSAGEConv(in_features, out_features, **kwargs)
        elif conv_type == 'gat':
            self.conv = GATConv(in_features, out_features, **kwargs)
        elif conv_type == 'gcn':
            self.conv = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
            
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, graph_data: Dict) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, n_nodes, in_features]
            graph_data: 图数据字典
            
        Returns:
            out: 输出特征 [batch, n_nodes, out_features]
        """
        if self.conv_type in ['cheb', 'sage', 'gat']:
            if self.conv_type == 'cheb':
                out = self.conv(x, graph_data['laplacian'])
            else:
                out = self.conv(x, graph_data['adj_matrix'])
        else:  # gcn
            out = self.conv(x)
            
        out = self.activation(out)
        out = self.norm(out)
        out = self.dropout(out)
        
        return out


class GCNNetwork(nn.Module):
    """完整的GCN网络"""
    
    def __init__(self, 
                 n_nodes: int,
                 n_features: int,
                 n_hidden: int = 64,
                 n_layers: int = 3,
                 conv_type: str = 'cheb',
                 prediction_steps: int = 1,
                 use_attention: bool = True,
                 use_dynamic_adj: bool = True,
                 dropout: float = 0.1):
        """
        初始化GCN网络
        
        Args:
            n_nodes: 节点数
            n_features: 输入特征维度
            n_hidden: 隐藏层维度
            n_layers: GCN层数
            conv_type: 卷积类型
            prediction_steps: 预测步数
            use_attention: 是否使用注意力机制
            use_dynamic_adj: 是否使用动态邻接矩阵
            dropout: dropout概率
        """
        super(GCNNetwork, self).__init__()
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.conv_type = conv_type
        self.prediction_steps = prediction_steps
        self.use_attention = use_attention
        self.use_dynamic_adj = use_dynamic_adj
        self.dropout = dropout
        
        # 输入层
        self.input_projection = nn.Linear(n_features, n_hidden)
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_features = n_hidden
            else:
                in_features = n_hidden
            out_features = n_hidden
            self.gcn_layers.append(
                GCNBlock(conv_type, in_features, out_features, dropout=dropout)
            )
        
        # 动态邻接矩阵学习器
        if use_dynamic_adj:
            self.dynamic_adj_learner = DynamicAdjacencyLearner(
                n_nodes, hidden_dim=n_hidden
            )
        
        # 多头注意力层
        if use_attention:
            self.attention = MultiHeadAttention(
                embed_dim=n_hidden,
                num_heads=8,
                dropout=dropout
            )
        
        # 输出层
        self.output_layers = nn.ModuleList()
        for step in range(prediction_steps):
            self.output_layers.append(
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(n_hidden // 2, n_features)
                )
            )
            
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, graph_data: Dict) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, seq_len, n_nodes, n_features]
            graph_data: 图数据字典
            
        Returns:
            outputs: 预测结果字典
        """
        batch_size, seq_len, n_nodes, n_features = x.shape
        
        # 重塑输入
        x = x.view(batch_size * seq_len, n_nodes, n_features)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 动态邻接矩阵学习
        if self.use_dynamic_adj:
            dynamic_adj = self.dynamic_adj_learner(x)
            # 更新图数据
            graph_data = graph_data.copy()
            graph_data['adj_matrix'] = dynamic_adj
            if 'laplacian' in graph_data:
                # 重新计算拉普拉斯矩阵
                laplacian = []
                for i in range(batch_size * seq_len):
                    adj = dynamic_adj[i]
                    degree = torch.diag(torch.sum(adj, dim=1))
                    lap = degree - adj
                    laplacian.append(lap)
                graph_data['laplacian'] = torch.stack(laplacian)
        
        # GCN层前向传播
        h = x
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, graph_data)
        
        # 注意力机制
        attention_weights = None
        if self.use_attention:
            # 重塑为序列形式
            h_reshaped = h.view(batch_size, seq_len, n_nodes, self.n_hidden)
            h_reshaped = h_reshaped.transpose(1, 2)  # [batch, n_nodes, seq_len, n_hidden]
            h_reshaped = h_reshaped.reshape(batch_size * n_nodes, seq_len, self.n_hidden)
            
            h_attended, attention_weights = self.attention(h_reshaped)
            
            # 重塑回原始形状
            h_attended = h_attended.view(batch_size, n_nodes, seq_len, self.n_hidden)
            h_attended = h_attended.transpose(1, 2)  # [batch, seq_len, n_nodes, n_hidden]
            h = h_attended.reshape(batch_size * seq_len, n_nodes, self.n_hidden)
        
        # 重塑为序列形式进行预测
        h = h.view(batch_size, seq_len, n_nodes, self.n_hidden)
        
        # 多步预测
        predictions = {}
        current_input = h[:, -1:]  # 使用最后一个时间步
        
        for step in range(self.prediction_steps):
            # 展平当前输入
            current_flat = current_input.view(-1, n_nodes, self.n_hidden)
            
            # 预测下一步
            pred = self.output_layers[step](current_flat)
            
            # 重塑并存储预测结果
            pred = pred.view(batch_size, 1, n_nodes, self.n_features)
            predictions[f'pred_{step}'] = pred
            
            # 更新输入（用于下一步预测）
            if step < self.prediction_steps - 1:
                # 简单复制预测结果作为下一步输入（在实际应用中可能需要更复杂的策略）
                current_input = pred
        
        # 合并所有预测结果
        final_predictions = torch.cat([predictions[f'pred_{i}'] for i in range(self.prediction_steps)], dim=1)
        
        outputs = {
            'predictions': final_predictions,  # [batch, prediction_steps, n_nodes, n_features]
            'hidden_states': h.view(batch_size, seq_len, n_nodes, self.n_hidden),
            'attention_weights': attention_weights,
            'dynamic_adjacency': dynamic_adj if self.use_dynamic_adj else None
        }
        
        return outputs


class GCNTrainer:
    """GCN网络训练器"""
    
    def __init__(self, 
                 model: GCNNetwork,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        初始化训练器
        
        Args:
            model: GCN网络模型
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, graph_data: Dict) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_size = batch_x.size(0)
            
            # 前向传播
            outputs = self.model(batch_x, graph_data)
            predictions = outputs['predictions']
            
            # 计算损失
            loss = self.criterion(predictions, batch_y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader, graph_data: Dict) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self.model(batch_x, graph_data)
                predictions = outputs['predictions']
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, 
              train_loader, 
              val_loader, 
              graph_data: Dict,
              epochs: int = 100,
              patience: int = 20) -> Dict:
        """训练模型"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, graph_data)
            val_loss = self.validate(val_loader, graph_data)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            training_history['learning_rates'].append(current_lr)
            
            # 更新学习率调度器
            self.scheduler.step(val_loss)
            
            # 记录损失
            training_history['train_losses'].append(train_loss)
            training_history['val_losses'].append(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_gcn_model.pth')
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, '
                      f'Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}')
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
                
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_gcn_model.pth'))
        
        return training_history
    
    def predict(self, x: torch.Tensor, graph_data: Dict) -> torch.Tensor:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x, graph_data)
            predictions = outputs['predictions']
        return predictions


class GCNEvaluator:
    """GCN网络评估器"""
    
    def __init__(self, model: GCNNetwork):
        """初始化评估器"""
        self.model = model
        
    def evaluate(self, test_loader, graph_data: Dict) -> Dict:
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                predictions = self.model.predict(batch_x, graph_data)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
        # 合并所有批次
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算指标
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """计算评估指标"""
        # 展平数组
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        
        metrics = {}
        
        # MAE
        metrics['MAE'] = mean_absolute_error(target_flat, pred_flat)
        
        # RMSE
        metrics['RMSE'] = np.sqrt(mean_squared_error(target_flat, pred_flat))
        
        # MAPE (避免除零)
        mask = np.abs(target_flat) > 1e-8
        if np.sum(mask) > 0:
            metrics['MAPE'] = np.mean(np.abs((target_flat[mask] - pred_flat[mask]) / target_flat[mask])) * 100
        else:
            metrics['MAPE'] = 0.0
            
        # R²
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        metrics['R2'] = 1 - (ss_res / (ss_tot + 1e-8))
        
        return metrics
    
    def plot_results(self, results: Dict, save_path: str = None):
        """可视化结果"""
        predictions = results['predictions']
        targets = results['targets']
        metrics = results['metrics']
        
        n_samples = min(100, predictions.shape[0])  # 最多显示100个样本
        n_steps = predictions.shape[1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 预测 vs 真实值散点图
        ax1 = axes[0, 0]
        pred_flat = predictions[:n_samples].reshape(-1)
        target_flat = targets[:n_samples].reshape(-1)
        ax1.scatter(target_flat, pred_flat, alpha=0.6, s=1)
        ax1.plot([target_flat.min(), target_flat.max()], 
                [target_flat.min(), target_flat.max()], 'r--', lw=2)
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title('预测 vs 真实值')
        
        # 2. 时间序列预测示例
        ax2 = axes[0, 1]
        sample_idx = 0
        node_idx = 0
        time_steps = range(n_steps)
        ax2.plot(time_steps, targets[sample_idx, :, node_idx, 0], 'b-', label='真实值', linewidth=2)
        ax2.plot(time_steps, predictions[sample_idx, :, node_idx, 0], 'r--', label='预测值', linewidth=2)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('值')
        ax2.set_title(f'时间序列预测示例 (样本={sample_idx}, 节点={node_idx})')
        ax2.legend()
        
        # 3. 损失曲线
        ax3 = axes[1, 0]
        # 这里需要从训练历史中获取损失曲线
        ax3.set_title('训练损失曲线')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        
        # 4. 指标条形图
        ax4 = axes[1, 1]
        metric_names = ['MAE', 'RMSE', 'MAPE', 'R2']
        metric_values = [metrics[m] for m in metric_names]
        bars = ax4.bar(metric_names, metric_values)
        ax4.set_title('评估指标')
        ax4.set_ylabel('值')
        
        # 在条形图上显示数值
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 打印指标
        print("\n=== 模型评估结果 ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")


def create_sample_data(n_timesteps: int = 1000, 
                      n_nodes: int = 50, 
                      n_features: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """创建示例数据"""
    # 生成节点坐标（随机分布在2D空间）
    np.random.seed(42)
    coordinates = np.random.uniform(0, 100, (n_nodes, 2))
    
    # 生成时序数据（模拟交通流数据）
    time = np.arange(n_timesteps)
    data = np.zeros((n_timesteps, n_nodes, n_features))
    
    for i in range(n_nodes):
        # 为每个节点生成具有周期性和趋势的时序数据
        base_trend = 50 + 10 * np.sin(2 * np.pi * time / 100)  # 日周期
        weekly_trend = 5 * np.sin(2 * np.pi * time / 700)  # 周周期
        noise = np.random.normal(0, 2, n_timesteps)
        
        data[:, i, 0] = base_trend + weekly_trend + noise
    
    return data, coordinates


def main():
    """主函数 - 演示GCN网络的使用"""
    print("=== GCN图卷积网络演示 ===\n")
    
    # 1. 创建示例数据
    print("1. 创建示例数据...")
    data, coordinates = create_sample_data(n_timesteps=500, n_nodes=30, n_features=1)
    print(f"数据形状: {data.shape}")
    print(f"坐标形状: {coordinates.shape}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    processor = GraphDataProcessor(normalization='zscore')
    graph_data = processor.prepare_graph_data(
        data, coordinates, window_size=12, prediction_steps=3
    )
    
    print(f"输入数据形状: {graph_data['X'].shape}")
    print(f"目标数据形状: {graph_data['y'].shape}")
    print(f"邻接矩阵形状: {graph_data['adj_matrix'].shape}")
    
    # 3. 数据集划分
    print("\n3. 数据集划分...")
    X = graph_data['X']
    y = graph_data['y']
    
    n_samples = X.shape[0]
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 创建模型
    print("\n4. 创建GCN模型...")
    model = GCNNetwork(
        n_nodes=graph_data['n_nodes'],
        n_features=graph_data['n_features'],
        n_hidden=64,
        n_layers=3,
        conv_type='cheb',  # 可选: 'cheb', 'sage', 'gat', 'gcn'
        prediction_steps=graph_data['prediction_steps'],
        use_attention=True,
        use_dynamic_adj=True,
        dropout=0.1
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 5. 训练模型
    print("\n5. 训练模型...")
    trainer = GCNTrainer(model, learning_rate=0.001)
    
    # 准备图数据（转换为张量）
    graph_data_tensor = {
        'adj_matrix': torch.FloatTensor(graph_data['adj_matrix']),
        'laplacian': torch.FloatTensor(graph_data['laplacian']),
        'coordinates': torch.FloatTensor(graph_data['coordinates'])
    }
    
    training_history = trainer.train(
        train_loader, val_loader, graph_data_tensor, 
        epochs=50, patience=15
    )
    
    # 6. 评估模型
    print("\n6. 评估模型...")
    evaluator = GCNEvaluator(model)
    results = evaluator.evaluate(test_loader, graph_data_tensor)
    
    # 7. 可视化结果
    print("\n7. 可视化结果...")
    evaluator.plot_results(results, save_path='gcn_results.png')
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()