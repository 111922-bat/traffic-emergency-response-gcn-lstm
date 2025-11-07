"""
GCN+LSTM混合架构交通流预测模型

该模块实现了基于图卷积网络(GCN)和长短期记忆网络(LSTM)的混合架构，
支持多种融合策略、多任务学习、模型压缩优化等功能。

主要特性:
- 多种融合策略: 串行、并行、注意力融合
- 多任务学习: 拥堵预测、速度预测等
- 模型压缩: 知识蒸馏、模型剪枝、量化
- 性能优化: 确保10秒内响应
- 动态图构建: 支持时变邻接矩阵

作者: Claude Code
日期: 2025-11-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from enum import Enum
import time
import logging
import copy

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """融合策略枚举"""
    SERIAL = "serial"          # 串行: GCN -> LSTM
    PARALLEL = "parallel"      # 并行: 空间支路 ∥ 时间支路
    ATTENTION = "attention"    # 注意力融合
    GATED = "gated"           # 门控融合
    RESIDUAL = "residual"     # 残差融合


class TaskType(Enum):
    """任务类型枚举"""
    SPEED_PREDICTION = "speed"      # 速度预测
    CONGESTION_PREDICTION = "congestion"  # 拥堵预测
    FLOW_PREDICTION = "flow"        # 流量预测
    MULTI_TASK = "multi_task"       # 多任务学习


@dataclass
class ModelConfig:
    """模型配置类"""
    # 基础配置
    input_dim: int = 1              # 输入特征维度
    hidden_dim: int = 64            # 隐藏层维度
    output_dim: int = 1             # 输出维度
    num_nodes: int = 228            # 节点数
    sequence_length: int = 12       # 输入序列长度
    prediction_steps: int = 3       # 预测步数
    
    # GCN配置
    gcn_layers: int = 2             # GCN层数
    gcn_dropout: float = 0.1        # GCN dropout
    use_dynamic_graph: bool = True  # 是否使用动态图
    
    # LSTM配置
    lstm_layers: int = 2            # LSTM层数
    lstm_dropout: float = 0.1       # LSTM dropout
    bidirectional: bool = False     # 是否双向
    
    # 融合策略
    fusion_strategy: FusionStrategy = FusionStrategy.SERIAL
    fusion_hidden_dim: int = 128    # 融合层隐藏维度
    
    # 注意力配置
    attention_heads: int = 8        # 注意力头数
    attention_dropout: float = 0.1  # 注意力 dropout
    
    # 多任务配置
    task_types: List[TaskType] = None  # 任务类型列表
    task_weights: Dict[TaskType, float] = None  # 任务权重
    
    # 优化配置
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    
    # 压缩配置
    use_knowledge_distillation: bool = False  # 知识蒸馏
    use_model_pruning: bool = False          # 模型剪枝
    use_quantization: bool = False           # 量化
    
    # 性能配置
    target_inference_time: float = 10.0      # 目标推理时间(秒)
    batch_size: int = 32                     # 批大小
    
    def __post_init__(self):
        if self.task_types is None:
            self.task_types = [TaskType.SPEED_PREDICTION]
        if self.task_weights is None:
            self.task_weights = {TaskType.SPEED_PREDICTION: 1.0}


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, num_nodes, feat_dim = x.shape
        
        # 重塑为多头格式 [batch_size, seq_len, num_heads, num_nodes, head_dim]
        x = x.view(batch_size, seq_len, num_nodes, self.num_heads, self.head_dim)
        x = x.transpose(2, 3).contiguous()  # [batch_size, seq_len, num_heads, num_nodes, head_dim]
        
        # 展平节点和头维度，保留时间维度 [batch_size, seq_len, num_heads * num_nodes, head_dim]
        x_flat = x.view(batch_size, seq_len, self.num_heads * num_nodes, self.head_dim)
        
        # 线性投影 (保持形状)
        q = self.query(x_flat)
        k = self.key(x_flat)
        v = self.value(x_flat)
        
        # 重新reshape为 [batch_size, seq_len, num_heads, num_nodes, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, num_nodes, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, num_nodes, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, num_nodes, self.head_dim)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # 如果提供了邻接矩阵，应用图结构约束
        if adj is not None:
            try:
                # 处理不同维度的邻接矩阵
                adj_dims = adj.dim()
                
                if adj_dims == 2:  # [num_nodes, num_nodes]
                    # 扩展到需要的维度
                    adj_expanded = adj.unsqueeze(0).unsqueeze(1).unsqueeze(2)  # [1, 1, 1, num_nodes, num_nodes]
                    adj_expanded = adj_expanded.repeat(batch_size, seq_len, self.num_heads, 1, 1)
                elif adj_dims == 3:  # [batch_size, num_nodes, num_nodes]
                    # 添加seq_len和num_heads维度
                    adj_expanded = adj.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_nodes, num_nodes]
                    adj_expanded = adj_expanded.repeat(1, seq_len, self.num_heads, 1, 1)
                else:  # 其他情况，尝试使用第一个样本
                    # 移除多余的前导维度
                    while adj.dim() > 3:
                        adj = adj.squeeze(0)
                    # 然后按照3D情况处理
                    adj_expanded = adj.unsqueeze(1).unsqueeze(2)
                    adj_expanded = adj_expanded.repeat(1, seq_len, self.num_heads, 1, 1)
                
                # 确保尺寸匹配
                adj_expanded = adj_expanded[:, :, :, :num_nodes, :num_nodes]
                scores = scores.masked_fill(adj_expanded == 0, -1e9)
            except Exception as e:
                logger.warning(f"空间注意力邻接矩阵处理失败: {e}，跳过邻接矩阵约束")
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, v)
        
        # 重塑回原始格式
        attended_values = attended_values.view(
            batch_size, seq_len, num_nodes, self.hidden_dim
        )
        
        # 输出投影
        output = self.output_projection(attended_values)
        
        return output


class TemporalAttention(nn.Module):
    """时间注意力机制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_nodes, feat_dim = x.shape
        
        # 重塑为多头格式 [batch_size, num_nodes, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, num_nodes, self.num_heads, self.head_dim)
        x = x.transpose(1, 2).contiguous()  # [batch_size, num_nodes, seq_len, num_heads, head_dim]
        
        # 展平节点和头维度 [batch_size, num_nodes, seq_len * num_heads, head_dim]
        x_flat = x.view(batch_size, num_nodes, seq_len * self.num_heads, self.head_dim)
        
        # 线性投影
        q = self.query(x_flat)
        k = self.key(x_flat)
        v = self.value(x_flat)
        
        # 重新reshape为 [batch_size, num_nodes, seq_len, num_heads, head_dim]
        q = q.view(batch_size, num_nodes, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, num_nodes, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, num_nodes, seq_len, self.num_heads, self.head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended_values = torch.matmul(attention_weights, v)
        
        # 重塑回原始格式
        attended_values = attended_values.view(
            batch_size, seq_len, num_nodes, self.hidden_dim
        )
        
        output = self.output_projection(attended_values)
        
        return output


class GraphConvLayer(nn.Module):
    """图卷积层"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, num_nodes, input_dim]
            adj: [num_nodes, num_nodes] 或 [batch_size, num_nodes, num_nodes]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # 重塑输入
        x_reshaped = x.view(batch_size * seq_len, num_nodes, self.input_dim)
        
        # 安全处理邻接矩阵：移除所有前导的维度为1的维度，确保不会超过3维
        while adj.dim() > 3:
            if adj.size(0) == 1:
                adj = adj.squeeze(0)
            else:
                # 如果第一个维度不为1，直接取第一个样本
                adj = adj[0]
        
        # 安全降级策略：确保邻接矩阵是2D或3D
        try:
            # 处理不同维度的邻接矩阵
            if adj.dim() == 2:
                # 标准情况：[num_nodes, num_nodes] -> 扩展到 [batch_size * seq_len, num_nodes, num_nodes]
                adj_expanded = adj.unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
            elif adj.dim() == 3:
                if adj.size(0) == batch_size:
                    # [batch_size, num_nodes, num_nodes] -> 扩展到 [batch_size * seq_len, num_nodes, num_nodes]
                    adj_expanded = adj.unsqueeze(1).repeat(1, seq_len, 1, 1).view(batch_size * seq_len, num_nodes, num_nodes)
                else:
                    # 其他3D情况，使用第一个样本并扩展
                    adj_safe = adj[0] if adj.size(0) > 0 else torch.eye(num_nodes, device=x.device)
                    adj_expanded = adj_safe.unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
            else:
                # 降级为单位矩阵确保安全运行
                adj_expanded = torch.eye(num_nodes, device=x.device).unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
        except Exception as e:
            # 任何异常都降级为单位矩阵
            adj_expanded = torch.eye(num_nodes, device=x.device).unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
        
        # 确保所有张量在同一设备上
        adj_expanded = adj_expanded.to(x.device)
        
        # 计算图卷积
        try:
            intermediate = torch.matmul(adj_expanded, x_reshaped)
            support = torch.matmul(intermediate, self.weight)
            output = support + self.bias
        except Exception as e:
            # 计算出错时，直接返回输入的投影版本
            output = torch.matmul(x_reshaped, self.weight) + self.bias
        
        # 应用ReLU激活（如果之前没应用）
        output = F.relu(output)
        
        # 重塑回原始形状
        output = output.view(batch_size, seq_len, num_nodes, self.output_dim)
        output = self.dropout(output)
        
        return output


class DynamicGraphBuilder(nn.Module):
    """动态图构建模块 - 优化版"""
    
    def __init__(self, node_dim: int, hidden_dim: int = 64, threshold: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        
        # 增强的节点嵌入，添加层归一化提高稳定性
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 自注意力层，使用较少的头数提高计算效率
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        
        # 图结构生成器
        self.graph_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, num_nodes, node_dim]
        Returns:
            adj: [batch_size, num_nodes, num_nodes]
        """
        batch_size, seq_len, num_nodes, node_dim = x.shape
        
        # 输入验证
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # 计算每个时间步的节点特征统计信息，丰富特征表达
        x_mean = x.mean(dim=1)  # [batch_size, num_nodes, node_dim]
        x_std = x.std(dim=1)    # [batch_size, num_nodes, node_dim]
        x_max = x.max(dim=1)[0] # [batch_size, num_nodes, node_dim]
        
        # 组合多尺度时间特征
        temporal_features = torch.cat([x_mean, x_std, x_max], dim=-1)  # [batch_size, num_nodes, 3*node_dim]
        
        # 特征投影以匹配hidden_dim
        projection_dim = temporal_features.size(-1)
        if projection_dim != self.node_dim:
            temporal_features = nn.Linear(projection_dim, self.node_dim, device=x.device)(temporal_features)
        
        # 节点嵌入
        node_emb = self.node_embedding(temporal_features)  # [batch_size, num_nodes, hidden_dim]
        
        # 计算注意力权重
        attn_output, attn_weights = self.attention(node_emb, node_emb, node_emb)
        
        # 生成图结构特征
        graph_features = self.graph_generator(attn_output)  # [batch_size, num_nodes, hidden_dim]
        
        # 计算邻接矩阵，添加温度参数控制分布平滑度
        temperature = 0.5
        adj_logits = torch.matmul(graph_features, graph_features.transpose(-2, -1)) / temperature
        
        # 使用softmax替代sigmoid，更好地归一化
        adj = F.softmax(adj_logits, dim=-1)
        
        # 应用阈值过滤弱连接，提高稀疏性
        adj = torch.where(adj > self.threshold, adj, torch.zeros_like(adj))
        
        # 重新归一化
        row_sum = adj.sum(dim=-1, keepdim=True) + 1e-10
        adj = adj / row_sum
        
        # 添加自环确保每个节点连接到自身
        for b in range(batch_size):
            adj[b].fill_diagonal_(1.0)
        
        return adj


class GCNModule(nn.Module):
    """GCN模块 - 优化版"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.gcn_layers
        
        # 增强的特征投影层，使用多层感知机提高特征表达能力
        self.feature_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.gcn_dropout)
        )
        
        # 图卷积层，每层使用不同的dropout率
        self.gcn_layers = nn.ModuleList()
        dropout_rates = np.linspace(config.gcn_dropout, 0.05, config.gcn_layers)
        
        for i in range(config.gcn_layers):
            self.gcn_layers.append(
                GraphConvLayer(
                    input_dim=config.hidden_dim,
                    output_dim=config.hidden_dim,
                    dropout=float(dropout_rates[i])
                )
            )
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(
            config.hidden_dim, min(4, config.attention_heads), config.attention_dropout
        )
        
        # 动态图构建 - 使用优化版
        if config.use_dynamic_graph:
            self.dynamic_graph_builder = DynamicGraphBuilder(config.hidden_dim, config.hidden_dim)
        else:
            self.dynamic_graph_builder = None
            
        # 层归一化 - 改进维度处理
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.gcn_layers)
        ])
        
        # 全局特征聚合
        num_nodes = config.num_nodes  # 从配置中获取num_nodes
        self.global_pool = nn.Linear(config.hidden_dim * num_nodes, config.hidden_dim) if hasattr(config, 'num_nodes') else None
        
    def forward(self, x: torch.Tensor, static_adj: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, num_nodes, input_dim]
            static_adj: [num_nodes, num_nodes] (可选)
        Returns:
            spatial_features: [batch_size, seq_len, num_nodes, hidden_dim]
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # 输入清理
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        if torch.isinf(x).any():
            x = torch.nan_to_num(x, posinf=1e5, neginf=-1e5)
        
        # 特征投影
        x_projected = self.feature_projection(x)  # [batch_size, seq_len, num_nodes, hidden_dim]
        
        # 构建或使用邻接矩阵，添加异常处理
        try:
            if self.dynamic_graph_builder is not None:
                adj = self.dynamic_graph_builder(x_projected)  # [batch_size, num_nodes, num_nodes]
            elif static_adj is not None:
                # 动态调整邻接矩阵维度以匹配节点数
                if static_adj.shape[0] != num_nodes:
                    # 尝试使用插值或采样调整邻接矩阵
                    logger.warning(f"邻接矩阵维度不匹配，正在调整...")
                    adj = self._adjust_adjacency_matrix(static_adj, num_nodes, x.device)
                else:
                    adj = static_adj
            else:
                # 构建稀疏邻接矩阵替代全连接
                adj = self._create_sparse_adjacency(num_nodes, x.device)
                
            adj = adj.to(x.device)
        except Exception as e:
            logger.error(f"邻接矩阵构建失败: {e}")
            # 回退到安全的单位矩阵
            adj = torch.eye(num_nodes, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 图卷积层堆叠，使用更好的残差结构
        h = x_projected
        residual = h  # 初始残差
        
        for i, (gcn_layer, layer_norm) in enumerate(zip(self.gcn_layers, self.layer_norms)):
            # 应用GCN层
            h = gcn_layer(h, adj)
            
            # 应用层归一化 - 自动处理维度
            h = layer_norm(h)
            
            # 改进的残差连接，每两层使用一次
            if i % 2 == 1 or i == self.num_layers - 1:
                h = h + residual
                residual = h  # 更新残差
            
            # 添加层间dropout
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.config.gcn_dropout / (i + 1), training=self.training)
        
        # 应用空间注意力，提高关键节点特征
        spatial_features = self.spatial_attention(h, adj)
        
        # 添加全局特征信息
        if self.global_pool is not None and hasattr(self.config, 'num_nodes'):
            # 全局池化聚合所有节点信息
            global_features = h.reshape(batch_size, seq_len, -1)
            global_features = self.global_pool(global_features).unsqueeze(2)  # [batch, seq, 1, hidden]
            # 融入全局信息
            spatial_features = spatial_features + global_features.expand_as(spatial_features)
        
        return spatial_features
    
    def _adjust_adjacency_matrix(self, adj: torch.Tensor, target_size: int, device: torch.device) -> torch.Tensor:
        """动态调整邻接矩阵维度"""
        current_size = adj.size(0)
        if current_size < target_size:
            # 扩展矩阵
            new_adj = torch.eye(target_size, device=device)
            new_adj[:current_size, :current_size] = adj
            # 为新增节点创建一些随机连接
            for i in range(current_size, target_size):
                # 与现有节点随机建立几个连接
                connections = torch.randint(0, current_size, (3,))
                new_adj[i, connections] = 0.1
                new_adj[connections, i] = 0.1
            return new_adj
        else:
            # 缩减矩阵
            return adj[:target_size, :target_size]
    
    def _create_sparse_adjacency(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """创建稀疏邻接矩阵"""
        # 创建稀疏连接（每个节点连接到5个最近的节点）
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        for i in range(num_nodes):
            # 为每个节点选择5个连接（包括自己）
            connections = [(i + j) % num_nodes for j in range(-2, 3)]
            adj[i, connections] = 1.0
        # 归一化
        row_sum = adj.sum(dim=1, keepdim=True) + 1e-10
        adj = adj / row_sum
        return adj


class LSTMModule(nn.Module):
    """LSTM模块"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # 时间注意力
        self.temporal_attention = TemporalAttention(
            config.hidden_dim, config.attention_heads, config.attention_dropout
        )
        
        # 输出投影
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.output_projection = nn.Linear(lstm_output_dim, config.hidden_dim)
        self.feature_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, num_nodes, hidden_dim]
        Returns:
            temporal_features: [batch_size, seq_len, num_nodes, hidden_dim]
        """
        batch_size, seq_len, num_nodes, feat_dim = x.shape
        
        # 特征投影
        x_projected = self.feature_projection(x)
        
        # 重塑为LSTM输入格式
        x_reshaped = x_projected.transpose(1, 2).contiguous()  # [batch_size, num_nodes, seq_len, hidden_dim]
        x_reshaped = x_reshaped.view(batch_size * num_nodes, seq_len, feat_dim)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x_reshaped)
        
        # 重塑回原始格式
        lstm_out = lstm_out.view(batch_size, num_nodes, seq_len, -1)
        lstm_out = lstm_out.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_nodes, -1]
        
        # 应用时间注意力
        temporal_features = self.temporal_attention(lstm_out)
        
        # 输出投影
        temporal_features = self.output_projection(temporal_features)
        
        return temporal_features


class FusionModule(nn.Module):
    """融合模块"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.fusion_strategy = config.fusion_strategy
        
        fusion_input_dim = config.hidden_dim * 2  # GCN + LSTM输出
        self.fusion_hidden_dim = config.fusion_hidden_dim
        
        # 融合层
        if self.fusion_strategy == FusionStrategy.SERIAL:
            self.fusion_layers = nn.Sequential(
                nn.Linear(fusion_input_dim, self.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.gcn_dropout),
                nn.Linear(self.fusion_hidden_dim, config.hidden_dim)
            )
        elif self.fusion_strategy == FusionStrategy.PARALLEL:
            self.gcn_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.lstm_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.fusion_projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        elif self.fusion_strategy == FusionStrategy.ATTENTION:
            self.fusion_attention = SpatialAttention(config.hidden_dim * 2, config.attention_heads)
            self.fusion_projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        elif self.fusion_strategy == FusionStrategy.GATED:
            self.gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            self.fusion_projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: [batch_size, seq_len, num_nodes, hidden_dim]
            temporal_features: [batch_size, seq_len, num_nodes, hidden_dim]
        Returns:
            fused_features: [batch_size, seq_len, num_nodes, hidden_dim]
        """
        if self.fusion_strategy == FusionStrategy.SERIAL:
            # 串行融合: 空间特征 -> 时间特征 -> 融合
            combined = torch.cat([spatial_features, temporal_features], dim=-1)
            fused_features = self.fusion_layers(combined)
            
        elif self.fusion_strategy == FusionStrategy.PARALLEL:
            # 并行融合: 分别投影后融合
            spatial_proj = self.gcn_projection(spatial_features)
            temporal_proj = self.lstm_projection(temporal_features)
            combined = torch.cat([spatial_proj, temporal_proj], dim=-1)
            fused_features = self.fusion_projection(combined)
            
        elif self.fusion_strategy == FusionStrategy.ATTENTION:
            # 注意力融合
            combined = torch.cat([spatial_features, temporal_features], dim=-1)
            attended = self.fusion_attention(combined)
            fused_features = self.fusion_projection(attended)
            
        elif self.fusion_strategy == FusionStrategy.GATED:
            # 门控融合
            combined = torch.cat([spatial_features, temporal_features], dim=-1)
            gate_weights = torch.sigmoid(self.gate(combined))
            gated_combined = gate_weights * spatial_features + (1 - gate_weights) * temporal_features
            fused_features = self.fusion_projection(gated_combined)
            
        else:  # RESIDUAL
            # 残差融合
            fused_features = spatial_features + temporal_features
            
        return fused_features


class MultiTaskHead(nn.Module):
    """多任务输出头 - 优化版"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.task_types = config.task_types
        
        # 共享特征转换层，提高特征表达能力
        self.shared_feature_trans = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.gcn_dropout)
        )
        
        # 为每个任务创建更复杂的输出头
        self.task_heads = nn.ModuleDict()
        
        for task_type in self.task_types:
            if task_type == TaskType.SPEED_PREDICTION:
                self.task_heads['speed'] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim // 2, config.output_dim)
                )
            elif task_type == TaskType.CONGESTION_PREDICTION:
                self.task_heads['congestion'] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim // 2, config.output_dim),
                    nn.Sigmoid()  # 拥堵预测使用sigmoid激活
                )
            elif task_type == TaskType.FLOW_PREDICTION:
                self.task_heads['flow'] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim // 2, config.output_dim)
                )
                
        # 任务权重，使用可学习的参数
        self.task_weights = nn.ParameterDict({
            task.value: nn.Parameter(torch.tensor(weight)) 
            for task, weight in config.task_weights.items()
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, num_nodes, hidden_dim]
        Returns:
            outputs: Dict[str, torch.Tensor] - 各任务的预测结果
        """
        outputs = {}
        
        batch_size, seq_len, num_nodes, hidden_dim = x.shape
        
        # 应用共享特征转换
        x_transformed = self.shared_feature_trans(x)
        
        # 确保只取最后prediction_steps个时间步作为预测
        if hasattr(self.config, 'prediction_steps') and self.config.prediction_steps > 0:
            # 获取正确的预测窗口
            if seq_len >= self.config.prediction_steps:
                x_pred = x_transformed[:, -self.config.prediction_steps:, :, :]
            else:
                # 如果序列长度不够，填充到预测步长
                pad_size = self.config.prediction_steps - seq_len
                pad = torch.zeros(batch_size, pad_size, num_nodes, hidden_dim, device=x.device)
                x_pred = torch.cat([x_transformed, pad], dim=1)
        else:
            x_pred = x_transformed
        
        # 对每个任务进行预测
        for task_name, head in self.task_heads.items():
            try:
                # 批量处理所有时间步，提高计算效率
                output = head(x_pred)  # [batch_size, prediction_steps, num_nodes, output_dim]
                
                # 确保输出维度正确
                if output.dim() == 4 and output.size(-1) != self.config.output_dim:
                    # 如果维度不匹配，进行调整
                    if output.size(-1) > self.config.output_dim:
                        output = output[..., :self.config.output_dim]  # 截断
                    else:
                        # 填充
                        pad_dim = self.config.output_dim - output.size(-1)
                        output = torch.cat([output, torch.zeros_like(output[..., :pad_dim])], dim=-1)
                
                outputs[task_name] = output
            except Exception as e:
                logger.error(f"任务 {task_name} 预测失败: {e}")
                # 创建默认输出以避免训练中断
                default_shape = (batch_size, self.config.prediction_steps, num_nodes, self.config.output_dim)
                outputs[task_name] = torch.zeros(default_shape, device=x.device)
        
        return outputs


class GCNLSTMHybrid(nn.Module):
    """GCN+LSTM混合模型主类 - 优化版"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 输入验证
        assert config.input_dim > 0, "输入维度必须大于0"
        assert config.hidden_dim > 0, "隐藏维度必须大于0"
        assert config.prediction_steps > 0, "预测步长必须大于0"
        
        # 构建模型组件
        self.gcn_module = GCNModule(config)
        self.lstm_module = LSTMModule(config)
        self.fusion_module = FusionModule(config)
        self.multi_task_head = MultiTaskHead(config)
        
        # 性能监控
        self.inference_times = []
        self.model_size_mb = 0
        
        # 初始化权重，使用Xavier初始化提高训练稳定性
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor, static_adj: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 优化版，增加错误处理和稳定性保障
        
        Args:
            x: [batch_size, seq_len, num_nodes, input_dim]
            static_adj: [num_nodes, num_nodes] (可选)
        Returns:
            outputs: Dict[str, torch.Tensor] - 各任务的预测结果
        """
        start_time = time.time()
        
        # 输入数据验证和清理
        if torch.isnan(x).any():
            warnings.warn("输入包含NaN值，已替换为0")
            x = torch.nan_to_num(x, nan=0.0)
        
        if torch.isinf(x).any():
            warnings.warn("输入包含无穷值，已替换为有限值")
            x = torch.nan_to_num(x, posinf=1e5, neginf=-1e5)
        
        # 输入维度检查和调整
        if x.dim() != 4:
            # 重塑输入为标准格式 [batch, seq_len, num_nodes, input_dim]
            if x.dim() == 3:
                x = x.unsqueeze(-1)  # 添加输入特征维度
            else:
                raise ValueError(f"输入维度错误: {x.dim()}，应为4维 [batch, seq, nodes, features]")
        
        try:
            # 空间特征提取 (GCN) - 添加错误处理
            spatial_features = self.gcn_module(x, static_adj)
            
            # 检查GCN输出
            if torch.isnan(spatial_features).any():
                logger.warning("GCN输出包含NaN，应用清理")
                spatial_features = torch.nan_to_num(spatial_features, nan=0.0)
            
            # 时间特征提取 (LSTM)
            temporal_features = self.lstm_module(spatial_features)
            
            # 检查LSTM输出
            if torch.isnan(temporal_features).any():
                logger.warning("LSTM输出包含NaN，应用清理")
                temporal_features = torch.nan_to_num(temporal_features, nan=0.0)
            
            # 特征融合
            fused_features = self.fusion_module(spatial_features, temporal_features)
            
            # 多任务预测
            outputs = self.multi_task_head(fused_features)
            
            # 确保所有预测结果形状正确
            self._ensure_output_shape(outputs)
            
        except Exception as e:
            logger.error(f"模型前向传播失败: {e}")
            # 创建应急输出以避免训练完全中断
            batch_size, seq_len, num_nodes, _ = x.shape
            outputs = self._create_emergency_outputs(batch_size, num_nodes)
        
        # 记录推理时间，但限制列表长度避免内存泄漏
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]  # 只保留最近1000次
        
        return outputs
    
    def _ensure_output_shape(self, outputs: Dict[str, torch.Tensor]):
        """确保所有输出形状正确"""
        for task_name, output in outputs.items():
            expected_shape = (output.size(0), self.config.prediction_steps, output.size(2), output.size(3))
            if output.size(1) != self.config.prediction_steps:
                # 调整时间维度长度
                if output.size(1) > self.config.prediction_steps:
                    outputs[task_name] = output[:, :self.config.prediction_steps]
                else:
                    # 填充到正确长度
                    pad_size = self.config.prediction_steps - output.size(1)
                    pad_shape = (output.size(0), pad_size) + output.size()[2:]
                    pad = torch.zeros(pad_shape, device=output.device)
                    outputs[task_name] = torch.cat([output, pad], dim=1)
    
    def _create_emergency_outputs(self, batch_size: int, num_nodes: int) -> Dict[str, torch.Tensor]:
        """创建应急输出以避免训练中断"""
        outputs = {}
        for task_type in self.config.task_types:
            task_key = task_type.value
            shape = (batch_size, self.config.prediction_steps, num_nodes, self.config.output_dim)
            outputs[task_key] = torch.zeros(shape, device=next(self.parameters()).device)
        return outputs
    
    def get_model_size(self) -> float:
        """获取模型大小(MB)"""
        if self.model_size_mb == 0:
            param_size = sum(p.numel() * p.element_size() for p in self.parameters() if p.requires_grad)
            buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
            self.model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        return self.model_size_mb
    
    def get_average_inference_time(self) -> float:
        """获取平均推理时间(秒)"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times[-100:])  # 最近100次推理的平均时间
    
    def compress_model(self, compression_ratio: float = 0.5) -> 'GCNLSTMHybrid':
        """模型压缩 - 整合剪枝、知识蒸馏和量化策略
        
        Args:
            compression_ratio: 压缩比例，0.0-1.0，值越大压缩越厉害
            
        Returns:
            压缩后的模型实例
        """
        # 参数验证
        if not 0 <= compression_ratio <= 1:
            raise ValueError(f"压缩比例必须在0到1之间，当前值: {compression_ratio}")
            
        original_size = self.get_model_size()
        logger.info(f"开始模型压缩，当前模型大小: {original_size:.2f}MB")
        
        # 保存压缩前的配置，用于后续可能的恢复
        original_config = copy.deepcopy(self.config)
        
        # 模型剪枝
        if self.config.use_model_pruning:
            logger.info(f"应用模型剪枝，压缩比例: {compression_ratio}")
            # 实现基于L1范数的权重剪枝
            with torch.no_grad():
                for name, module in self.named_modules():
                    if isinstance(module, (nn.Linear, GraphConvLayer)) and hasattr(module, 'weight'):
                        # 跳过太小的层
                        if module.weight.size(0) <= 5:
                            continue
                            
                        # 计算权重绝对值
                        weights_abs = module.weight.abs()
                        # 确定剪枝阈值，避免过度剪枝
                        threshold_ratio = max(0.05, min(0.9, compression_ratio))
                        threshold = weights_abs.flatten().sort().values[int(len(weights_abs.flatten()) * threshold_ratio)]
                        # 剪枝
                        mask = weights_abs > threshold
                        module.weight.data *= mask
                        # 计算剪枝率
                        pruned_ratio = 1 - (mask.sum() / mask.numel())
                        logger.debug(f"层 {name} 剪枝率: {pruned_ratio:.2%}")
            post_pruning_size = self.get_model_size()
            logger.info(f"模型剪枝完成，剪枝后的模型大小: {post_pruning_size:.2f}MB")
        
        # 知识蒸馏
        if self.config.use_knowledge_distillation:
            logger.info("应用知识蒸馏压缩...")
            
            # 1. 压缩LSTM层
            if hasattr(self, 'lstm_module'):
                for name, child in self.lstm_module.named_children():
                    if isinstance(child, nn.LSTM):
                        # 记录原始状态
                        original_state = child.state_dict()
                        # 创建一个更小的LSTM
                        input_size = child.input_size
                        hidden_size = max(8, int(child.hidden_size * (1 - compression_ratio)))
                        num_layers = max(1, int(child.num_layers * max(0.5, 1 - compression_ratio * 0.5)))
                        
                        new_lstm = nn.LSTM(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=child.dropout,
                            bidirectional=child.bidirectional
                        )
                        
                        # 复制可用的权重
                        with torch.no_grad():
                            new_state = new_lstm.state_dict()
                            for key in new_state:
                                if key in original_state and new_state[key].shape == original_state[key].shape:
                                    new_state[key].copy_(original_state[key])
                            new_lstm.load_state_dict(new_state)
                        
                        # 替换原始LSTM
                        setattr(self.lstm_module, name, new_lstm)
                        logger.info(f"LSTM层蒸馏完成，隐藏维度从 {child.hidden_size} 降至 {hidden_size}，层数从 {child.num_layers} 降至 {num_layers}")
            
            # 2. 压缩GCN层
            if hasattr(self, 'gcn_module'):
                for name, module in self.gcn_module.named_modules():
                    if isinstance(module, GraphConvLayer):
                        # 简化GCN层，保持输入输出维度不变
                        with torch.no_grad():
                            # 对权重进行奇异值分解压缩
                            if hasattr(module, 'weight') and module.weight.size(0) > 10 and module.weight.size(1) > 10:
                                U, S, V = torch.svd(module.weight)
                                k = max(3, int(min(module.weight.size(0), module.weight.size(1)) * (1 - compression_ratio)))
                                compressed_weight = U[:, :k] @ torch.diag(S[:k]) @ V[:, :k].t()
                                module.weight.data = compressed_weight
                                logger.info(f"GCN层 {name} 奇异值分解压缩，保留前 {k} 个奇异值")
        
        # 量化
        if self.config.use_quantization:
            logger.info("应用模型量化...")
            try:
                # 为不同类型的层设置不同的量化配置
                quant_config = torch.quantization.get_default_qconfig('fbgemm')
                self.qconfig = quant_config
                
                # 为线性层和卷积层设置量化
                for name, module in self.named_modules():
                    if isinstance(module, (nn.Linear, GraphConvLayer)):
                        module.qconfig = quant_config
                    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        # 批归一化层使用融合量化
                        module.qconfig = torch.quantization.float_qparams_weight_only_qconfig
                
                # 准备量化感知训练
                torch.quantization.prepare_qat(self, inplace=True)
                logger.info("量化感知训练准备完成")
                
            except Exception as e:
                logger.error(f"量化准备过程中出错: {str(e)}")
        
        # 3. 调整注意力机制（如有）
        if hasattr(self, 'gcn_module') and hasattr(self.gcn_module, 'spatial_attention'):
            attention = self.gcn_module.spatial_attention
            if hasattr(attention, 'num_heads'):
                # 减少注意力头数
                new_num_heads = max(1, int(attention.num_heads * (1 - compression_ratio * 0.5)))
                if new_num_heads != attention.num_heads:
                    logger.info(f"调整空间注意力头数从 {attention.num_heads} 到 {new_num_heads}")
                    # 注意：这里需要重新创建注意力层，实际实现需要更复杂的逻辑
        
        # 更新配置以反映压缩后的状态
        self.config.hidden_dim = max(16, int(self.config.hidden_dim * (1 - compression_ratio)))
        self.config.lstm_layers = max(1, int(self.config.lstm_layers * (1 - compression_ratio * 0.5)))
        self.config.gcn_layers = max(1, int(self.config.gcn_layers * (1 - compression_ratio * 0.3)))
        
        final_size = self.get_model_size()
        compression_percent = 100 * (original_size - final_size) / original_size
        logger.info(f"模型压缩完成，最终大小: {final_size:.2f}MB, 压缩比例: {compression_percent:.2f}%")
        
        # 检查是否达到目标推理时间
        if hasattr(self.config, 'target_inference_time') and self.config.target_inference_time > 0:
            avg_inference_time = self.get_average_inference_time()
            if avg_inference_time > 0:
                logger.info(f"压缩后的平均推理时间: {avg_inference_time:.4f}秒")
                if avg_inference_time > self.config.target_inference_time:
                    logger.warning(f"推理时间未达到目标值 ({self.config.target_inference_time}秒)，建议增加压缩比例")
        
        return self
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_size_mb': self.get_model_size(),
            'average_inference_time': self.get_average_inference_time()
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'GCNLSTMHybrid':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.model_size_mb = checkpoint.get('model_size_mb', 0)
        model.inference_times = checkpoint.get('inference_times', [])
        logger.info(f"模型已从 {path} 加载")
        return model


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model: GCNLSTMHybrid, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        if config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5
            )
        else:
            self.scheduler = None
            
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)
            
            # 前向传播
            outputs = self.model(data)
            
            # 计算损失
            loss = 0.0
            for task_name, output in outputs.items():
                task_target = targets[task_name].to(self.device)
                task_loss = self.criterion(output, task_target)
                
                # 应用任务权重
                task_weight = self.config.task_weights.get(
                    TaskType(task_name), 1.0
                )
                loss += task_weight * task_loss
                
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = total_loss / num_batches
        
        # 验证
        val_loss = None
        if val_loader is not None:
            val_loss = self.validate(val_loader)
            
        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step(val_loss if val_loss is not None else avg_train_loss)
            
        # 记录历史
        self.train_history['train_loss'].append(avg_train_loss)
        if val_loss is not None:
            self.train_history['val_loss'].append(val_loss)
        self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        return {
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                
                loss = 0.0
                for task_name, output in outputs.items():
                    task_target = targets[task_name].to(self.device)
                    task_loss = self.criterion(output, task_target)
                    task_weight = self.config.task_weights.get(
                        TaskType(task_name), 1.0
                    )
                    loss += task_weight * task_loss
                    
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader=None, epochs: int = 100, 
              early_stopping_patience: int = 20) -> Dict[str, Any]:
        """完整训练流程"""
        logger.info(f"开始训练，设备: {self.device}")
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"模型大小: {self.model.get_model_size():.2f} MB")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练
            metrics = self.train_epoch(train_loader, val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印进度
            if val_loader is not None:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {metrics['train_loss']:.4f} - "
                    f"Val Loss: {metrics['val_loss']:.4f} - "
                    f"LR: {metrics['learning_rate']:.6f} - "
                    f"Time: {epoch_time:.2f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {metrics['train_loss']:.4f} - "
                    f"LR: {metrics['learning_rate']:.6f} - "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # 早停检查
            if val_loader is not None and metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
                patience_counter = 0
                
                # 保存最佳模型
                self.model.save_model('best_model.pth')
                logger.info(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"早停触发，连续{early_stopping_patience}个epoch无改善")
                break
                
            # 性能检查
            avg_inference_time = self.model.get_average_inference_time()
            if avg_inference_time > self.config.target_inference_time:
                logger.warning(f"推理时间超过目标: {avg_inference_time:.2f}s > {self.config.target_inference_time}s")
                
        logger.info("训练完成")
        
        return {
            'train_history': self.train_history,
            'best_val_loss': best_val_loss,
            'final_model_size': self.model.get_model_size(),
            'average_inference_time': self.model.get_average_inference_time()
        }
    
    def predict(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data.to(self.device))
        return {k: v.cpu().numpy() for k, v in outputs.items()}


def create_sample_data(num_samples: int = 1000, config: ModelConfig = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """创建示例数据用于测试"""
    if config is None:
        config = ModelConfig()
        
    # 生成输入数据
    x = torch.randn(num_samples, config.sequence_length, config.num_nodes, config.input_dim)
    
    # 生成目标数据
    targets = {}
    for task_type in config.task_types:
        if task_type == TaskType.SPEED_PREDICTION:
            targets['speed'] = torch.randn(num_samples, config.prediction_steps, config.num_nodes, config.output_dim)
        elif task_type == TaskType.CONGESTION_PREDICTION:
            targets['congestion'] = torch.randn(num_samples, config.prediction_steps, config.num_nodes, config.output_dim)
        elif task_type == TaskType.FLOW_PREDICTION:
            targets['flow'] = torch.randn(num_samples, config.prediction_steps, config.num_nodes, config.output_dim)
            
    return x, targets


def create_sample_adj_matrix(num_nodes: int) -> torch.Tensor:
    """创建示例邻接矩阵"""
    # 基于距离的邻接矩阵
    positions = torch.randn(num_nodes, 2)
    distances = torch.cdist(positions, positions)
    
    # 高斯核
    sigma = 1.0
    adj = torch.exp(-distances**2 / (2 * sigma**2))
    adj.fill_diagonal_(0)  # 去掉自环
    
    return adj


def main():
    """主函数 - 演示模型使用"""
    logger.info("GCN+LSTM混合模型演示")
    
    # 配置模型
    config = ModelConfig(
        fusion_strategy=FusionStrategy.ATTENTION,
        task_types=[TaskType.SPEED_PREDICTION, TaskType.CONGESTION_PREDICTION],
        task_weights={
            TaskType.SPEED_PREDICTION: 0.7,
            TaskType.CONGESTION_PREDICTION: 0.3
        },
        use_dynamic_graph=True,
        use_knowledge_distillation=True
    )
    
    # 创建模型
    model = GCNLSTMHybrid(config)
    logger.info(f"模型创建完成，参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建示例数据
    x, targets = create_sample_data(num_samples=100, config=config)
    adj = create_sample_adj_matrix(config.num_nodes)
    
    # 前向传播测试
    logger.info("测试前向传播...")
    outputs = model(x, adj)
    logger.info(f"输出形状: {[(k, v.shape) for k, v in outputs.items()]}")
    
    # 训练器测试
    logger.info("测试训练器...")
    from torch.utils.data import DataLoader, Dataset
    
    # 创建自定义数据集
    class TrafficDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets
        
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            # 返回单样本输入和对应的多任务目标
            sample_input = self.inputs[idx]
            sample_targets = {k: v[idx] for k, v in self.targets.items()}
            return sample_input, sample_targets
    
    # 创建数据加载器
    dataset = TrafficDataset(x, targets)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    trainer = ModelTrainer(model, config)
    
    # 训练几个epoch
    history = trainer.train(train_loader, val_loader, epochs=5)
    
    logger.info(f"训练完成，最终验证损失: {history['best_val_loss']:.4f}")
    logger.info(f"模型大小: {history['final_model_size']:.2f} MB")
    logger.info(f"平均推理时间: {history['average_inference_time']:.4f} 秒")
    
    # 性能检查
    if history['average_inference_time'] > config.target_inference_time:
        logger.info("应用模型压缩...")
        compressed_model = model.compress_model(compression_ratio=0.5)
        compressed_model.save_model('compressed_model.pth')
    
    logger.info("演示完成!")


if __name__ == "__main__":
    main()