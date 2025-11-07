"""
拥堵扩散预测算法实现
基于物理模型和深度学习的混合架构，支持30分钟内拥堵扩散预测

主要功能：
1. 基于物理模型的拥堵传播计算
2. 结合深度学习模型进行预测
3. 多时间步拥堵扩散预测
4. 瓶颈路段识别和影响范围计算
5. 应急情况下的拥堵控制策略
6. 实时预测和批量预测支持

作者：TrafficAI Team
日期：2025-11-05
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from collections import deque
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import time
from dataclasses import dataclass
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CongestionLevel(Enum):
    """拥堵等级枚举"""
    FREE_FLOW = 0  # 自由流
    LIGHT = 1      # 轻度拥堵
    MODERATE = 2   # 中度拥堵
    HEAVY = 3      # 重度拥堵
    SEVERE = 4     # 严重拥堵


class PredictionMode(Enum):
    """预测模式枚举"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    EMERGENCY = "emergency"


@dataclass
class RoadSegment:
    """道路路段数据结构"""
    segment_id: str
    length: float  # 长度 (km)
    lanes: int     # 车道数
    capacity: int  # 通行能力 (veh/h)
    free_flow_speed: float  # 自由流速度 (km/h)
    current_speed: float    # 当前速度 (km/h)
    current_flow: int       # 当前流量 (veh/h)
    occupancy: float        # 占有率 (0-1)
    bottleneck_score: float = 0.0  # 瓶颈评分
    
    @property
    def congestion_level(self) -> CongestionLevel:
        """根据速度计算拥堵等级"""
        speed_ratio = self.current_speed / self.free_flow_speed
        if speed_ratio >= 0.9:
            return CongestionLevel.FREE_FLOW
        elif speed_ratio >= 0.7:
            return CongestionLevel.LIGHT
        elif speed_ratio >= 0.5:
            return CongestionLevel.MODERATE
        elif speed_ratio >= 0.3:
            return CongestionLevel.HEAVY
        else:
            return CongestionLevel.SEVERE
    
    @property
    def v_c_ratio(self) -> float:
        """计算v/c比"""
        return self.current_flow / self.capacity if self.capacity > 0 else 0


@dataclass
class PredictionResult:
    """预测结果数据结构"""
    segment_id: str
    predicted_speeds: np.ndarray
    predicted_flows: np.ndarray
    predicted_occupancy: np.ndarray
    congestion_levels: List[CongestionLevel]
    confidence_scores: np.ndarray
    propagation_speed: float
    influence_range: float
    timestamp: float


class FundamentalDiagram:
    """基本图模型类"""
    
    def __init__(self, model_type: str = "triangular"):
        """
        初始化基本图模型
        
        Args:
            model_type: 基本图类型 ('linear', 'log', 'exponential', 'triangular')
        """
        self.model_type = model_type
        self.params = {}
    
    def set_parameters(self, v_f: float, k_jam: float, q_max: float = None, 
                      w: float = None, v_m: float = None):
        """
        设置基本图参数
        
        Args:
            v_f: 自由流速度 (km/h)
            k_jam: 堵塞密度 (veh/km)
            q_max: 最大流量 (veh/h)
            w: 拥堵波速度 (km/h)
            v_m: 最佳速度 (km/h)
        """
        self.params = {
            'v_f': v_f,
            'k_jam': k_jam,
            'q_max': q_max,
            'w': w,
            'v_m': v_m
        }
    
    def speed_density_relationship(self, k: np.ndarray) -> np.ndarray:
        """速度-密度关系"""
        if self.model_type == "linear":
            return self.params['v_f'] * (1 - k / self.params['k_jam'])
        elif self.model_type == "log":
            return self.params['v_m'] * np.log(self.params['k_jam'] / np.maximum(k, 1e-8))
        elif self.model_type == "exponential":
            return self.params['v_f'] * np.exp(-k / self.params['k_jam'])
        elif self.model_type == "triangular":
            k_c = self.params['q_max'] / self.params['v_f']  # 临界密度
            v = np.where(k <= k_c, 
                        self.params['v_f'], 
                        self.params['v_f'] * (1 - k / self.params['k_jam']))
            return np.maximum(v, 0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def flow_density_relationship(self, k: np.ndarray) -> np.ndarray:
        """流量-密度关系"""
        v = self.speed_density_relationship(k)
        return v * k
    
    def shockwave_speed(self, k1: float, k2: float) -> float:
        """计算冲击波速度"""
        q1 = self.flow_density_relationship(np.array([k1]))[0]
        q2 = self.flow_density_relationship(np.array([k2]))[0]
        return (q2 - q1) / (k2 - k1) if k2 != k1 else 0


class CellTransmissionModel:
    """元胞传输模型 (CTM)"""
    
    def __init__(self, segments: List[RoadSegment], dt: float = 0.1):
        """
        初始化CTM
        
        Args:
            segments: 道路路段列表
            dt: 时间步长 (小时)
        """
        self.segments = segments
        self.dt = dt
        self.n_segments = len(segments)
        self.fd = FundamentalDiagram("triangular")
        
        # 初始化状态变量
        self.densities = np.zeros(self.n_segments)
        self.flows = np.zeros(self.n_segments)
        self.speeds = np.zeros(self.n_segments)
        
        # 设置基本图参数
        self._initialize_fundamental_diagram()
    
    def _initialize_fundamental_diagram(self):
        """初始化基本图参数"""
        # 使用第一个路段的参数作为默认参数
        if self.segments:
            v_f = self.segments[0].free_flow_speed
            capacity = self.segments[0].capacity
            length = self.segments[0].length
            
            # 计算参数
            k_jam = 180  # veh/km (典型值)
            q_max = capacity
            w = 15  # km/h (拥堵波速度)
            
            self.fd.set_parameters(v_f, k_jam, q_max, w)
    
    def update_state(self, upstream_flow: float, downstream_density: float = 0):
        """更新CTM状态"""
        new_densities = np.zeros(self.n_segments)
        new_flows = np.zeros(self.n_segments)
        
        for i in range(self.n_segments):
            segment = self.segments[i]
            
            # 发送流量 (基于当前密度)
            k_send = min(self.densities[i], segment.capacity / segment.free_flow_speed)
            q_send = k_send * segment.free_flow_speed
            
            # 接收流量 (基于下游密度)
            if i < self.n_segments - 1:
                k_receive = segment.capacity / segment.free_flow_speed
            else:
                k_receive = k_jam if 'k_jam' in self.fd.params else 180
            
            q_receive = k_receive * segment.free_flow_speed
            
            # 实际流量
            actual_flow = min(q_send, q_receive)
            
            # 密度更新
            if i == 0:
                # 上游边界
                density_change = (upstream_flow - actual_flow) * self.dt / segment.length
            else:
                # 中间路段
                density_change = (new_flows[i-1] - actual_flow) * self.dt / segment.length
            
            new_densities[i] = max(0, self.densities[i] + density_change)
            new_flows[i] = actual_flow
        
        # 更新状态
        self.densities = new_densities
        self.flows = new_flows
        self.speeds = self.fd.speed_density_relationship(self.densities)
        
        return self.speeds, self.flows, self.densities
    
    def simulate_propagation(self, initial_congestion: int, duration: float = 0.5) -> Dict:
        """
        模拟拥堵传播
        
        Args:
            initial_congestion: 初始拥堵路段索引
            duration: 仿真持续时间 (小时)
            
        Returns:
            仿真结果字典
        """
        steps = int(duration / self.dt)
        results = {
            'speeds': [],
            'flows': [],
            'densities': [],
            'congestion_fronts': []
        }
        
        # 设置初始拥堵
        self.densities[initial_congestion] = self.fd.params['k_jam'] * 0.7  # 70% 堵塞密度
        
        for step in range(steps):
            # 更新状态
            speeds, flows, densities = self.update_state(upstream_flow=1000)
            
            results['speeds'].append(speeds.copy())
            results['flows'].append(flows.copy())
            results['densities'].append(densities.copy())
            
            # 寻找拥堵前沿
            congestion_front = self._find_congestion_front(speeds)
            results['congestion_fronts'].append(congestion_front)
        
        return results
    
    def _find_congestion_front(self, speeds: np.ndarray) -> int:
        """寻找拥堵前沿位置"""
        speed_threshold = np.mean(speeds) * 0.7  # 速度阈值为平均速度的70%
        congested_indices = np.where(speeds < speed_threshold)[0]
        return int(congested_indices[-1]) if len(congested_indices) > 0 else -1


class DynamicGraphConstructor:
    """动态图构建器"""
    
    def __init__(self, n_nodes: int, embedding_dim: int = 64):
        """
        初始化动态图构建器
        
        Args:
            n_nodes: 节点数量
            embedding_dim: 嵌入维度
        """
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Parameter(torch.randn(n_nodes, embedding_dim))
        self.temporal_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        
    def construct_dynamic_graph(self, features: torch.Tensor, 
                              adjacency_mode: str = "attention") -> torch.Tensor:
        """
        构建动态图
        
        Args:
            features: 节点特征 [seq_len, n_nodes, feature_dim]
            adjacency_mode: 邻接矩阵构建模式 ('distance', 'attention', 'correlation')
            
        Returns:
            动态邻接矩阵 [seq_len, n_nodes, n_nodes]
        """
        # 处理不同的输入形状
        if len(features.shape) == 3:
            # [seq_len, n_nodes, feature_dim]
            batch_size, seq_len, n_nodes, feature_dim = 1, *features.shape
            features = features.unsqueeze(0)  # 添加batch维度
        else:
            # [batch_size, seq_len, n_nodes, feature_dim]
            batch_size, seq_len, n_nodes, feature_dim = features.shape
        
        if adjacency_mode == "distance":
            # 基于距离的邻接矩阵
            return self._distance_based_adjacency(features)
        elif adjacency_mode == "attention":
            # 基于注意力的邻接矩阵
            return self._attention_based_adjacency(features)
        elif adjacency_mode == "correlation":
            # 基于相关性的邻接矩阵
            return self._correlation_based_adjacency(features)
        else:
            raise ValueError(f"Unknown adjacency mode: {adjacency_mode}")
    
    def _distance_based_adjacency(self, features: torch.Tensor) -> torch.Tensor:
        """基于距离的邻接矩阵"""
        # 计算节点间的欧氏距离
        distances = torch.cdist(features, features, p=2)
        
        # 转换为邻接矩阵 (距离越小，连接越强)
        sigma = 1.0  # 距离尺度参数
        adjacency = torch.exp(-distances**2 / (2 * sigma**2))
        
        return adjacency
    
    def _attention_based_adjacency(self, features: torch.Tensor) -> torch.Tensor:
        """基于注意力的邻接矩阵"""
        batch_size, seq_len, n_nodes, feature_dim = features.shape
        adjacency = torch.zeros(seq_len, n_nodes, n_nodes)
        
        # 动态调整节点嵌入以匹配实际节点数
        if n_nodes != self.n_nodes:
            # 重新初始化节点嵌入
            self.node_embeddings = nn.Parameter(torch.randn(n_nodes, self.embedding_dim))
        
        for t in range(seq_len):
            # 使用节点嵌入计算注意力权重
            query = self.node_embeddings[:n_nodes, :]  # [n_nodes, embedding_dim]
            key = self.node_embeddings[:n_nodes, :]    # [n_nodes, embedding_dim]
            
            # 计算注意力权重 [n_nodes, n_nodes]
            attention_weights = torch.softmax(
                torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embedding_dim), 
                dim=-1
            )
            adjacency[t] = attention_weights
        
        return adjacency
    
    def _correlation_based_adjacency(self, features: torch.Tensor) -> torch.Tensor:
        """基于相关性的邻接矩阵"""
        batch_size, seq_len, n_nodes, feature_dim = features.shape
        adjacency = torch.zeros(seq_len, n_nodes, n_nodes)
        
        for t in range(seq_len):
            # 计算特征相关性
            feature_matrix = features[0, t]  # [n_nodes, feature_dim]
            correlation_matrix = torch.corrcoef(feature_matrix.transpose(0, 1))
            adjacency[t] = torch.abs(correlation_matrix)
        
        return adjacency


class SpatialTemporalGCN(nn.Module):
    """时空图卷积网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.1):
        """
        初始化时空GCN
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 层数
            dropout: Dropout比率
        """
        super(SpatialTemporalGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                gcn = GCNConv(input_dim, hidden_dim)
            elif i == num_layers - 1:
                gcn = GCNConv(hidden_dim, output_dim)
            else:
                gcn = GCNConv(hidden_dim, hidden_dim)
            self.gcn_layers.append(gcn)
        
        # 时间卷积层
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 残差连接
        self.residual = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [batch_size, seq_len, n_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edge_dim]
            
        Returns:
            输出特征 [batch_size, seq_len, n_nodes, output_dim]
        """
        # 处理不同的输入形状
        if len(x.shape) == 3:
            # [seq_len, n_nodes, input_dim] -> [1, seq_len, n_nodes, input_dim]
            batch_size, seq_len, n_nodes, input_dim = 1, *x.shape
            x = x.unsqueeze(0)
        else:
            # [batch_size, seq_len, n_nodes, input_dim]
            batch_size, seq_len, n_nodes, input_dim = x.shape
        
        # 变形为GCN输入格式
        x_reshaped = x.view(-1, n_nodes, input_dim)  # [batch_size * seq_len, n_nodes, input_dim]
        x_reshaped = x_reshaped.transpose(0, 1)  # [n_nodes, batch_size * seq_len, input_dim]
        
        # 图卷积
        for i, gcn_layer in enumerate(self.gcn_layers):
            if i == 0:
                residual = self.residual(x_reshaped.transpose(0, 1))  # [batch_size * seq_len, n_nodes, hidden_dim]
            
            # GCN前向传播
            x_reshaped = gcn_layer(x_reshaped, edge_index, edge_attr)
            x_reshaped = F.relu(x_reshaped)
            x_reshaped = self.dropout_layer(x_reshaped)
            
            # 残差连接
            if i == 0:
                x_reshaped = x_reshaped + residual.transpose(0, 1)
        
        # 恢复原始形状
        output = x_reshaped.transpose(0, 1)  # [batch_size * seq_len, n_nodes, output_dim]
        output = output.view(batch_size, seq_len, n_nodes, self.output_dim)
        
        return output


class LSTMPredictor(nn.Module):
    """LSTM预测器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.1, bidirectional: bool = True):
        """
        初始化LSTM预测器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
            bidirectional: 是否双向
        """
        super(LSTMPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           dropout=dropout, bidirectional=bidirectional, 
                           batch_first=True)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(hidden_dim * (2 if bidirectional else 1), 
                                             num_heads=8)
        
        # 输出层
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_layer = nn.Linear(lstm_output_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            预测输出 [batch_size, seq_len, input_dim]
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended_out = attended_out.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 输出层
        output = self.output_layer(attended_out)
        
        return output


class CongestionPropagationPredictor:
    """拥堵扩散预测器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化拥堵扩散预测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.scaler = StandardScaler()
        self.fd = FundamentalDiagram(config.get('fundamental_diagram_type', 'triangular'))
        self.ctm = None
        self.dynamic_graph = None
        self.gcn_model = None
        self.lstm_model = None
        
        # 状态管理
        self.historical_data = deque(maxlen=config.get('history_length', 100))
        self.prediction_cache = {}
        
        # 瓶颈识别参数
        self.bottleneck_threshold = config.get('bottleneck_threshold', 0.8)
        self.propagation_speed_threshold = config.get('propagation_speed_threshold', 10.0)
        
        # 应急控制参数
        self.emergency_response_time = config.get('emergency_response_time', 300)  # 5分钟
        self.control_strategies = {
            'routing': self._routing_strategy,
            'signal': self._signal_strategy,
            'capacity': self._capacity_strategy
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        input_dim = self.config.get('input_dim', 4)  # speed, flow, occupancy, v/c
        hidden_dim = self.config.get('hidden_dim', 64)
        output_dim = self.config.get('output_dim', 3)  # speed, flow, occupancy
        
        # 初始化GCN模型
        self.gcn_model = SpatialTemporalGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=self.config.get('gcn_layers', 3),
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)
        
        # 初始化LSTM模型
        self.lstm_model = LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=self.config.get('lstm_layers', 2),
            dropout=self.config.get('dropout', 0.1),
            bidirectional=self.config.get('bidirectional', True)
        ).to(self.device)
        
        # 初始化动态图构建器
        self.dynamic_graph = DynamicGraphConstructor(
            n_nodes=self.config.get('n_nodes', 50),
            embedding_dim=self.config.get('embedding_dim', 64)
        )
        
        logger.info(f"模型初始化完成，使用设备: {self.device}")
    
    def prepare_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        数据预处理
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            预处理后的张量
        """
        # 特征工程
        features = ['speed', 'flow', 'occupancy', 'v_c_ratio']
        
        if 'v_c_ratio' not in data.columns:
            data['v_c_ratio'] = data['flow'] / (data['capacity'] + 1e-8)
        
        # 填充缺失值
        data[features] = data[features].fillna(method='ffill').fillna(method='bfill')
        
        # 归一化
        scaled_data = self.scaler.fit_transform(data[features])
        
        # 转换为张量
        tensor_data = torch.FloatTensor(scaled_data).to(self.device)
        
        return tensor_data
    
    def build_road_network(self, segments: List[RoadSegment], 
                          adjacency_type: str = "distance") -> Data:
        """
        构建路网图结构
        
        Args:
            segments: 道路路段列表
            adjacency_type: 邻接矩阵类型
            
        Returns:
            PyTorch Geometric Data对象
        """
        n_nodes = len(segments)
        
        # 构建邻接矩阵
        positions = np.array([[i, 0] for i in range(n_nodes)])  # 简化为一维位置
        
        if adjacency_type == "distance":
            # 基于距离的邻接矩阵
            distances = pdist(positions)
            adjacency_matrix = squareform(distances)
            
            # 转换为边索引
            non_zero_distances = adjacency_matrix[adjacency_matrix > 0]
            if len(non_zero_distances) > 0:
                threshold = np.percentile(non_zero_distances, 50)  # 50%分位数作为阈值
            else:
                threshold = 1.0  # 默认阈值
            
            edge_index = []
            edge_attr = []
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and adjacency_matrix[i, j] <= threshold:
                        edge_index.append([i, j])
                        edge_attr.append([1.0 / (adjacency_matrix[i, j] + 1e-8)])  # 距离倒数作为权重
        
        else:
            # 简单的线性连接
            edge_index = []
            edge_attr = []
            
            for i in range(n_nodes - 1):
                edge_index.extend([[i, i + 1], [i + 1, i]])
                edge_attr.extend([[1.0], [1.0]])
            
            # 如果没有边（只有一个节点），添加自环
            if len(edge_index) == 0:
                edge_index = [[0, 0]]
                edge_attr = [[1.0]]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # 节点特征
        node_features = torch.tensor([
            [seg.current_speed, seg.current_flow, seg.occupancy, seg.v_c_ratio]
            for seg in segments
        ], dtype=torch.float)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def identify_bottlenecks(self, segments: List[RoadSegment]) -> List[str]:
        """
        识别瓶颈路段
        
        Args:
            segments: 道路路段列表
            
        Returns:
            瓶颈路段ID列表
        """
        bottlenecks = []
        
        for segment in segments:
            # 计算瓶颈评分
            bottleneck_score = (
                0.3 * segment.v_c_ratio +  # 流量比
                0.3 * (1 - segment.current_speed / segment.free_flow_speed) +  # 速度损失比
                0.2 * segment.occupancy +  # 占有率
                0.2 * self._calculate_propagation_risk(segment)  # 传播风险
            )
            
            segment.bottleneck_score = bottleneck_score
            
            if bottleneck_score >= self.bottleneck_threshold:
                bottlenecks.append(segment.segment_id)
        
        # 按瓶颈评分排序
        bottlenecks.sort(key=lambda x: next(
            seg.bottleneck_score for seg in segments if seg.segment_id == x
        ), reverse=True)
        
        logger.info(f"识别出 {len(bottlenecks)} 个瓶颈路段: {bottlenecks}")
        return bottlenecks
    
    def _calculate_propagation_risk(self, segment: RoadSegment) -> float:
        """计算传播风险"""
        # 基于路段特性的传播风险计算
        risk_factors = {
            'length': min(segment.length / 5.0, 1.0),  # 长度风险
            'lanes': min(segment.lanes / 4.0, 1.0),   # 车道数风险
            'capacity_utilization': min(segment.v_c_ratio, 2.0) / 2.0  # 容量利用率风险
        }
        
        return sum(risk_factors.values()) / len(risk_factors)
    
    def predict_congestion_propagation(self, segments: List[RoadSegment], 
                                     prediction_horizon: int = 6,
                                     mode: PredictionMode = PredictionMode.BATCH) -> List[PredictionResult]:
        """
        预测拥堵扩散
        
        Args:
            segments: 道路路段列表
            prediction_horizon: 预测时间步数 (每个时间步5分钟)
            mode: 预测模式
            
        Returns:
            预测结果列表
        """
        logger.info(f"开始拥堵扩散预测，预测步数: {prediction_horizon}, 模式: {mode.value}")
        
        # 识别瓶颈
        bottlenecks = self.identify_bottlenecks(segments)
        
        # 构建路网图
        graph_data = self.build_road_network(segments)
        
        # 准备输入数据
        input_data = self.prepare_data(pd.DataFrame([
            {
                'segment_id': seg.segment_id,
                'speed': seg.current_speed,
                'flow': seg.current_flow,
                'occupancy': seg.occupancy,
                'capacity': seg.capacity
            } for seg in segments
        ]))
        
        # 调整数据形状为 [batch_size, seq_len, n_nodes, features]
        batch_size = 1
        seq_len = min(self.config.get('input_sequence_length', 12), input_data.shape[0])
        input_tensor = input_data[-seq_len:].unsqueeze(0)  # [1, seq_len, n_nodes, features]
        
        # 构建动态图
        dynamic_adjacency = self.dynamic_graph.construct_dynamic_graph(
            input_tensor, adjacency_mode="attention"
        )
        
        # 深度学习预测
        with torch.no_grad():
            # GCN预测
            gcn_output = self.gcn_model(input_tensor, graph_data.edge_index, graph_data.edge_attr)
            
            # LSTM预测
            lstm_input = input_tensor.view(batch_size, seq_len, -1)  # [batch_size, seq_len, n_nodes*features]
            lstm_output = self.lstm_model(lstm_input)
            
            # 融合预测结果
            fused_output = self._fuse_predictions(gcn_output, lstm_output)
            
            # 多步预测
            predictions = self._multi_step_prediction(
                fused_output, dynamic_adjacency, graph_data, prediction_horizon
            )
        
        # 生成预测结果
        results = []
        for i, segment in enumerate(segments):
            pred_result = PredictionResult(
                segment_id=segment.segment_id,
                predicted_speeds=predictions['speeds'][:, i],
                predicted_flows=predictions['flows'][:, i],
                predicted_occupancy=predictions['occupancy'][:, i],
                congestion_levels=self._classify_congestion_levels(predictions['speeds'][:, i]),
                confidence_scores=predictions['confidence'][:, i],
                propagation_speed=self._calculate_propagation_speed(segment, predictions),
                influence_range=self._calculate_influence_range(segment, predictions),
                timestamp=time.time()
            )
            results.append(pred_result)
        
        logger.info(f"预测完成，生成 {len(results)} 个路段的预测结果")
        return results
    
    def _fuse_predictions(self, gcn_output: torch.Tensor, lstm_output: torch.Tensor) -> torch.Tensor:
        """融合GCN和LSTM预测结果"""
        # 调整LSTM输出形状以匹配GCN输出
        batch_size, seq_len, n_nodes, features = gcn_output.shape
        lstm_reshaped = lstm_output.view(batch_size, seq_len, n_nodes, features)
        
        # 加权融合
        alpha = self.config.get('fusion_weight', 0.6)  # GCN权重
        fused_output = alpha * gcn_output + (1 - alpha) * lstm_reshaped
        
        return fused_output
    
    def _multi_step_prediction(self, initial_output: torch.Tensor, 
                             adjacency: torch.Tensor, graph_data: Data, 
                             horizon: int) -> Dict[str, np.ndarray]:
        """多步预测"""
        batch_size, seq_len, n_nodes, features = initial_output.shape
        predictions = {
            'speeds': np.zeros((horizon, n_nodes)),
            'flows': np.zeros((horizon, n_nodes)),
            'occupancy': np.zeros((horizon, n_nodes)),
            'confidence': np.zeros((horizon, n_nodes))
        }
        
        current_output = initial_output.clone()
        
        for t in range(horizon):
            # 使用当前状态进行预测
            with torch.no_grad():
                # GCN前向传播
                gcn_pred = self.gcn_model(current_output, graph_data.edge_index, graph_data.edge_attr)
                
                # LSTM前向传播
                lstm_input = current_output.view(batch_size, seq_len, -1)
                lstm_pred = self.lstm_model(lstm_input)
                
                # 融合预测
                fused_pred = self._fuse_predictions(gcn_pred, lstm_pred)
                
                # 转换为物理量
                speeds = fused_pred[0, -1, :, 0].cpu().numpy()  # 速度
                flows = fused_pred[0, -1, :, 1].cpu().numpy()  # 流量
                occupancy = fused_pred[0, -1, :, 2].cpu().numpy()  # 占有率
                
                # 物理约束修正
                speeds, flows, occupancy = self._apply_physical_constraints(
                    speeds, flows, occupancy
                )
                
                # 计算置信度
                confidence = self._calculate_confidence(fused_pred)
                
                # 存储预测结果
                predictions['speeds'][t] = speeds
                predictions['flows'][t] = flows
                predictions['occupancy'][t] = occupancy
                predictions['confidence'][t] = confidence
            
            # 更新输入序列 (滑动窗口)
            if t < horizon - 1:
                # 将当前预测添加到序列中
                new_step = fused_pred[0, -1]  # [n_nodes, features]
                current_output = torch.cat([
                    current_output[0, 1:],  # 移除最旧的步骤
                    new_step.unsqueeze(0)   # 添加新步骤
                ], dim=0).unsqueeze(0)  # 恢复batch维度
        
        return predictions
    
    def _apply_physical_constraints(self, speeds: np.ndarray, flows: np.ndarray, 
                                  occupancy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """应用物理约束"""
        # 速度约束
        speeds = np.clip(speeds, 0, 120)  # 速度范围 0-120 km/h
        
        # 流量约束 (基于基本图)
        max_flows = self.fd.params.get('q_max', 2000)  # 最大流量
        flows = np.clip(flows, 0, max_flows)
        
        # 占有率约束
        occupancy = np.clip(occupancy, 0, 1)
        
        # 基本图一致性约束
        for i in range(len(speeds)):
            density = flows[i] / max(speeds[i], 1e-8)  # k = q/v
            theoretical_speed = self.fd.speed_density_relationship(np.array([density]))[0]
            
            # 如果速度与基本图不一致，进行修正
            if abs(speeds[i] - theoretical_speed) > 10:  # 误差超过10 km/h
                speeds[i] = 0.8 * speeds[i] + 0.2 * theoretical_speed
        
        return speeds, flows, occupancy
    
    def _calculate_confidence(self, prediction: torch.Tensor) -> np.ndarray:
        """计算预测置信度"""
        # 基于预测值的变化幅度计算置信度
        pred_std = torch.std(prediction, dim=-1).cpu().numpy()
        confidence = 1.0 / (1.0 + pred_std)  # 标准差越小，置信度越高
        return confidence[0] if len(confidence.shape) > 1 else confidence
    
    def _classify_congestion_levels(self, speeds: np.ndarray) -> List[CongestionLevel]:
        """分类拥堵等级"""
        # 使用默认自由流速度进行分类
        default_vf = 60  # km/h
        
        levels = []
        for speed in speeds:
            speed_ratio = speed / default_vf
            if speed_ratio >= 0.9:
                levels.append(CongestionLevel.FREE_FLOW)
            elif speed_ratio >= 0.7:
                levels.append(CongestionLevel.LIGHT)
            elif speed_ratio >= 0.5:
                levels.append(CongestionLevel.MODERATE)
            elif speed_ratio >= 0.3:
                levels.append(CongestionLevel.HEAVY)
            else:
                levels.append(CongestionLevel.SEVERE)
        
        return levels
    
    def _calculate_propagation_speed(self, segment: RoadSegment, predictions: Dict) -> float:
        """计算传播速度"""
        # 简化实现：基于速度变化趋势
        speeds = predictions['speeds'][:, 0]  # 假设第一个节点为代表
        
        if len(speeds) < 2:
            return 0.0
        
        # 计算速度变化率
        speed_change = np.diff(speeds)
        avg_change_rate = np.mean(speed_change)
        
        # 转换为传播速度 (km/h)
        propagation_speed = abs(avg_change_rate) * 60  # 假设时间步为1分钟
        
        return min(propagation_speed, self.propagation_speed_threshold)
    
    def _calculate_influence_range(self, segment: RoadSegment, predictions: Dict) -> float:
        """计算影响范围"""
        # 基于拥堵持续时间和传播速度计算影响范围
        speeds = predictions['speeds'][:, 0]
        
        # 识别拥堵时段
        congested_periods = speeds < (segment.free_flow_speed * 0.7)
        congestion_duration = np.sum(congested_periods) * 5  # 假设每步5分钟
        
        # 计算影响范围
        propagation_speed = self._calculate_propagation_speed(segment, predictions)
        influence_range = propagation_speed * (congestion_duration / 60)  # km
        
        return min(influence_range, 50.0)  # 最大影响范围50km
    
    def emergency_control_strategy(self, bottlenecks: List[str], 
                                 current_state: List[RoadSegment],
                                 strategy_type: str = 'routing') -> Dict[str, Any]:
        """
        应急控制策略
        
        Args:
            bottlenecks: 瓶颈路段列表
            current_state: 当前道路状态
            strategy_type: 策略类型 ('routing', 'signal', 'capacity')
            
        Returns:
            控制策略结果
        """
        logger.info(f"执行应急控制策略: {strategy_type}")
        
        if strategy_type not in self.control_strategies:
            raise ValueError(f"未知策略类型: {strategy_type}")
        
        strategy_func = self.control_strategies[strategy_type]
        result = strategy_func(bottlenecks, current_state)
        
        # 策略效果评估
        result['effectiveness'] = self._evaluate_strategy_effectiveness(result, current_state)
        result['implementation_time'] = self.emergency_response_time
        result['cost_estimate'] = self._estimate_strategy_cost(result)
        
        return result
    
    def _routing_strategy(self, bottlenecks: List[str], 
                        current_state: List[RoadSegment]) -> Dict[str, Any]:
        """路径诱导策略"""
        # 识别替代路径
        alternative_routes = self._find_alternative_routes(bottlenecks)
        
        # 计算路径诱导效果
        traffic_diversion = {}
        for bottleneck in bottlenecks:
            # 计算可转移的交通量
            segment = next((s for s in current_state if s.segment_id == bottleneck), None)
            if segment:
                diversion_rate = min(0.3, segment.v_c_ratio - 0.8)  # 最大30%转移率
                diverted_flow = segment.current_flow * diversion_rate
                traffic_diversion[bottleneck] = {
                    'diverted_flow': diverted_flow,
                    'alternative_routes': alternative_routes.get(bottleneck, [])
                }
        
        return {
            'strategy_type': 'routing',
            'target_segments': bottlenecks,
            'traffic_diversion': traffic_diversion,
            'implementation': 'dynamic_routing',
            'expected_delay_reduction': 0.25  # 预期延误减少25%
        }
    
    def _signal_strategy(self, bottlenecks: List[str], 
                        current_state: List[RoadSegment]) -> Dict[str, Any]:
        """信号控制策略"""
        # 信号优化参数
        signal_optimization = {}
        
        for bottleneck in bottlenecks:
            segment = next((s for s in current_state if s.segment_id == bottleneck), None)
            if segment:
                # 计算绿灯时间调整
                current_green_ratio = 0.6  # 假设当前绿灯比例
                target_green_ratio = min(0.8, current_green_ratio + 0.2 * (segment.v_c_ratio - 0.8))
                
                signal_optimization[bottleneck] = {
                    'current_green_ratio': current_green_ratio,
                    'target_green_ratio': target_green_ratio,
                    'cycle_adjustment': target_green_ratio - current_green_ratio
                }
        
        return {
            'strategy_type': 'signal',
            'target_segments': bottlenecks,
            'signal_optimization': signal_optimization,
            'implementation': 'adaptive_signal_control',
            'expected_capacity_increase': 0.15  # 预期通行能力增加15%
        }
    
    def _capacity_strategy(self, bottlenecks: List[str], 
                         current_state: List[RoadSegment]) -> Dict[str, Any]:
        """通行能力提升策略"""
        # 临时通行能力提升措施
        capacity_enhancement = {}
        
        for bottleneck in bottlenecks:
            segment = next((s for s in current_state if s.segment_id == bottleneck), None)
            if segment:
                # 计算临时车道开放效果
                temporary_lanes = min(2, max(0, int(segment.lanes * 0.5)))  # 最多开放一半车道
                capacity_increase = temporary_lanes * (segment.capacity / segment.lanes)
                
                capacity_enhancement[bottleneck] = {
                    'temporary_lanes': temporary_lanes,
                    'capacity_increase': capacity_increase,
                    'implementation_cost': temporary_lanes * 10000  # 每车道10万元成本
                }
        
        return {
            'strategy_type': 'capacity',
            'target_segments': bottlenecks,
            'capacity_enhancement': capacity_enhancement,
            'implementation': 'temporary_lane_opening',
            'expected_capacity_increase': 0.30  # 预期通行能力增加30%
        }
    
    def _find_alternative_routes(self, bottlenecks: List[str]) -> Dict[str, List[str]]:
        """寻找替代路径"""
        # 简化实现：返回预定义的替代路径
        alternative_routes = {}
        
        for bottleneck in bottlenecks:
            # 假设的替代路径
            alternative_routes[bottleneck] = [
                f"route_via_{bottleneck}_north",
                f"route_via_{bottleneck}_south"
            ]
        
        return alternative_routes
    
    def _evaluate_strategy_effectiveness(self, strategy_result: Dict, 
                                       current_state: List[RoadSegment]) -> float:
        """评估策略效果"""
        strategy_type = strategy_result['strategy_type']
        
        # 基于策略类型的预期效果
        effectiveness_map = {
            'routing': 0.25,
            'signal': 0.15,
            'capacity': 0.30
        }
        
        base_effectiveness = effectiveness_map.get(strategy_type, 0.1)
        
        # 根据瓶颈严重程度调整效果
        bottleneck_severity = np.mean([
            next((s.v_c_ratio for s in current_state if s.segment_id == bid), 0)
            for bid in strategy_result.get('target_segments', [])
        ])
        
        # 严重程度越高，策略效果越明显
        adjusted_effectiveness = base_effectiveness * (1 + bottleneck_severity - 0.8)
        
        return min(adjusted_effectiveness, 0.5)  # 最大效果50%
    
    def _estimate_strategy_cost(self, strategy_result: Dict) -> float:
        """估算策略成本"""
        strategy_type = strategy_result['strategy_type']
        
        cost_map = {
            'routing': 50000,      # 5万元 (系统部署)
            'signal': 100000,      # 10万元 (信号优化)
            'capacity': 200000     # 20万元 (临时措施)
        }
        
        base_cost = cost_map.get(strategy_type, 50000)
        
        # 根据目标路段数量调整成本
        target_count = len(strategy_result.get('target_segments', []))
        
        return base_cost * (1 + 0.2 * (target_count - 1))
    
    def real_time_prediction(self, current_data: Dict[str, Any], 
                           prediction_horizon: int = 6) -> PredictionResult:
        """
        实时预测
        
        Args:
            current_data: 当前数据
            prediction_horizon: 预测步数
            
        Returns:
            预测结果
        """
        # 构建路段对象
        segments = []
        for segment_id, data in current_data.items():
            segment = RoadSegment(
                segment_id=segment_id,
                length=data.get('length', 1.0),
                lanes=data.get('lanes', 3),
                capacity=data.get('capacity', 2000),
                free_flow_speed=data.get('free_flow_speed', 60),
                current_speed=data.get('current_speed', 50),
                current_flow=data.get('current_flow', 1500),
                occupancy=data.get('occupancy', 0.3)
            )
            segments.append(segment)
        
        # 执行预测
        results = self.predict_congestion_propagation(
            segments, prediction_horizon, PredictionMode.REAL_TIME
        )
        
        # 返回第一个路段的结果作为示例
        return results[0] if results else None
    
    def batch_prediction(self, historical_data: List[Dict], 
                        prediction_horizon: int = 6) -> List[PredictionResult]:
        """
        批量预测
        
        Args:
            historical_data: 历史数据列表
            prediction_horizon: 预测步数
            
        Returns:
            预测结果列表
        """
        all_results = []
        
        for data_point in historical_data:
            # 转换为DataFrame
            df = pd.DataFrame(data_point)
            
            # 构建路段对象
            segments = []
            for _, row in df.iterrows():
                segment = RoadSegment(
                    segment_id=row['segment_id'],
                    length=row['length'],
                    lanes=row['lanes'],
                    capacity=row['capacity'],
                    free_flow_speed=row['free_flow_speed'],
                    current_speed=row['current_speed'],
                    current_flow=row['current_flow'],
                    occupancy=row['occupancy']
                )
                segments.append(segment)
            
            # 执行预测
            results = self.predict_congestion_propagation(
                segments, prediction_horizon, PredictionMode.BATCH
            )
            
            all_results.extend(results)
        
        logger.info(f"批量预测完成，处理了 {len(historical_data)} 个时间点")
        return all_results
    
    def save_model(self, path: str):
        """保存模型"""
        model_state = {
            'gcn_state_dict': self.gcn_model.state_dict(),
            'lstm_state_dict': self.lstm_model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'fd_params': self.fd.params
        }
        
        torch.save(model_state, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.gcn_model.load_state_dict(checkpoint['gcn_state_dict'])
        self.lstm_model.load_state_dict(checkpoint['lstm_state_dict'])
        self.scaler = checkpoint['scaler']
        self.fd.params = checkpoint['fd_params']
        
        logger.info(f"模型已从 {path} 加载")


def create_sample_data(n_segments: int = 20, n_timesteps: int = 50) -> List[RoadSegment]:
    """创建示例数据"""
    segments = []
    
    for i in range(n_segments):
        segment = RoadSegment(
            segment_id=f"segment_{i:03d}",
            length=np.random.uniform(0.5, 3.0),
            lanes=np.random.randint(2, 6),
            capacity=np.random.randint(1500, 3000),
            free_flow_speed=np.random.uniform(50, 80),
            current_speed=np.random.uniform(30, 70),
            current_flow=np.random.randint(800, 2500),
            occupancy=np.random.uniform(0.2, 0.8)
        )
        segments.append(segment)
    
    return segments


def visualize_prediction_results(results: List[PredictionResult], 
                                save_path: str = None):
    """可视化预测结果"""
    if not results:
        logger.warning("没有预测结果可可视化")
        return
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 提取数据
    segment_ids = [r.segment_id for r in results]
    speeds = np.array([r.predicted_speeds for r in results])
    flows = np.array([r.predicted_flows for r in results])
    occupancy = np.array([r.predicted_occupancy for r in results])
    confidence = np.array([r.confidence_scores for r in results])
    
    # 速度预测
    im1 = axes[0, 0].imshow(speeds.T, aspect='auto', cmap='RdYlBu_r')
    axes[0, 0].set_title('速度预测 (km/h)')
    axes[0, 0].set_xlabel('路段')
    axes[0, 0].set_ylabel('时间步')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 流量预测
    im2 = axes[0, 1].imshow(flows.T, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('流量预测 (veh/h)')
    axes[0, 1].set_xlabel('路段')
    axes[0, 1].set_ylabel('时间步')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 占有率预测
    im3 = axes[1, 0].imshow(occupancy.T, aspect='auto', cmap='plasma')
    axes[1, 0].set_title('占有率预测')
    axes[1, 0].set_xlabel('路段')
    axes[1, 0].set_ylabel('时间步')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 置信度
    im4 = axes[1, 1].imshow(confidence.T, aspect='auto', cmap='Greens')
    axes[1, 1].set_title('预测置信度')
    axes[1, 1].set_xlabel('路段')
    axes[1, 1].set_ylabel('时间步')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存到: {save_path}")
    
    plt.show()


def evaluate_prediction_accuracy(results: List[PredictionResult], 
                               ground_truth: List[np.ndarray]) -> Dict[str, float]:
    """评估预测准确性"""
    if len(results) != len(ground_truth):
        raise ValueError("预测结果与真实值数量不匹配")
    
    metrics = {
        'mae_speed': [],
        'rmse_speed': [],
        'mae_flow': [],
        'rmse_flow': [],
        'mae_occupancy': [],
        'rmse_occupancy': []
    }
    
    for result, truth in zip(results, ground_truth):
        # 速度指标
        metrics['mae_speed'].append(mean_absolute_error(truth[:, 0], result.predicted_speeds))
        metrics['rmse_speed'].append(np.sqrt(mean_squared_error(truth[:, 0], result.predicted_speeds)))
        
        # 流量指标
        metrics['mae_flow'].append(mean_absolute_error(truth[:, 1], result.predicted_flows))
        metrics['rmse_flow'].append(np.sqrt(mean_squared_error(truth[:, 1], result.predicted_flows)))
        
        # 占有率指标
        metrics['mae_occupancy'].append(mean_absolute_error(truth[:, 2], result.predicted_occupancy))
        metrics['rmse_occupancy'].append(np.sqrt(mean_squared_error(truth[:, 2], result.predicted_occupancy)))
    
    # 计算平均值
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    
    return avg_metrics


if __name__ == "__main__":
    # 示例用法
    logger.info("拥堵扩散预测系统初始化")
    
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
        'bottleneck_threshold': 0.8,
        'propagation_speed_threshold': 10.0,
        'emergency_response_time': 300,
        'fundamental_diagram_type': 'triangular'
    }
    
    # 创建预测器
    predictor = CongestionPropagationPredictor(config)
    
    # 创建示例数据
    segments = create_sample_data(n_segments=20)
    
    # 执行预测
    logger.info("执行拥堵扩散预测")
    results = predictor.predict_congestion_propagation(
        segments, prediction_horizon=6, mode=PredictionMode.BATCH
    )
    
    # 识别瓶颈
    bottlenecks = predictor.identify_bottlenecks(segments)
    logger.info(f"识别出的瓶颈路段: {bottlenecks}")
    
    # 应急控制策略
    if bottlenecks:
        strategy_result = predictor.emergency_control_strategy(
            bottlenecks, segments, 'routing'
        )
        logger.info(f"应急策略结果: {strategy_result}")
    
    # 可视化结果
    visualize_prediction_results(results, '/workspace/code/models/prediction_results.png')
    
    # 保存模型
    predictor.save_model('/workspace/code/models/congestion_predictor.pth')
    
    logger.info("拥堵扩散预测系统演示完成")