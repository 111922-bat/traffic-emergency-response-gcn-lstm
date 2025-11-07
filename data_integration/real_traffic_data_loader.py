"""
真实交通数据加载器

此模块负责从data/real-data目录加载METR-LA和PEMS-BAY数据集，
并将其转换为GCN-LSTM模型所需的格式。

支持的功能：
- 加载CSV格式的交通数据集
- 数据预处理和标准化
- 构建时空序列数据
- 创建训练/验证/测试数据加载器
- 加载预定义的邻接矩阵
"""

import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, List, Dict, Optional
import logging
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTrafficDataLoader:
    """真实交通数据加载器类"""
    
    def __init__(self, data_dir: str = 'd:\gcn-lstm\data\real-data', dataset_name: str = 'METR-LA'):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
            dataset_name: 数据集名称，支持'METR-LA'或'PEMS-BAY'
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name.upper()
        self.data_path = os.path.join(data_dir, f"{self.dataset_name}.csv")
        self.meta_path = os.path.join(data_dir, f"{self.dataset_name}-META.csv") if self.dataset_name == 'PEMS-BAY' else None
        self.adj_path = os.path.join(data_dir, "adj_mx_bay.pkl") if self.dataset_name == 'PEMS-BAY' else None
        
        # 数据集特定参数
        if self.dataset_name == 'METR-LA':
            self.num_nodes = 207  # METR-LA有207个节点
            self.default_sequence_length = 12
            self.default_prediction_steps = 3
        elif self.dataset_name == 'PEMS-BAY':
            self.num_nodes = 325  # PEMS-BAY有325个节点
            self.default_sequence_length = 12
            self.default_prediction_steps = 3
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 数据存储
        self.raw_data = None
        self.adj_matrix = None
        self.scaler = StandardScaler()
        self.node_ids = None
        
        # 加载数据
        self.load_data()
        if self.meta_path and os.path.exists(self.meta_path):
            self.load_metadata()
        if self.adj_path and os.path.exists(self.adj_path) and self.dataset_name == 'PEMS-BAY':
            self.load_adj_matrix(self.adj_path)
        
        logger.info(f"成功初始化{self.dataset_name}数据加载器")
    
    def load_data(self):
        """加载CSV数据"""
        try:
            logger.info(f"加载数据: {self.data_path}")
            # 读取CSV文件，第一列是时间戳，其余列是节点数据
            self.raw_data = pd.read_csv(self.data_path)
            
            # 处理时间戳列
            if 'Unnamed: 0' in self.raw_data.columns:
                # 重命名时间戳列
                self.raw_data.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
                self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
                # 获取节点ID（列名）
                self.node_ids = [col for col in self.raw_data.columns if col != 'timestamp']
            else:
                # 假设第一列是时间戳
                self.raw_data.columns = ['timestamp'] + [f'node_{i}' for i in range(1, len(self.raw_data.columns))]
                self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
                self.node_ids = [col for col in self.raw_data.columns if col != 'timestamp']
            
            logger.info(f"数据加载完成，形状: {self.raw_data.shape}")
            logger.info(f"包含 {len(self.node_ids)} 个节点，{len(self.raw_data)} 个时间点")
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def load_metadata(self):
        """加载节点元数据"""
        try:
            logger.info(f"加载元数据: {self.meta_path}")
            self.metadata = pd.read_csv(self.meta_path)
            logger.info(f"元数据加载完成，包含 {len(self.metadata)} 个节点记录")
        except Exception as e:
            logger.warning(f"加载元数据失败: {e}")
            self.metadata = None
    
    def load_adj_matrix(self, path: str):
        """
        加载预定义的邻接矩阵
        
        Args:
            path: 邻接矩阵文件路径
        """
        try:
            logger.info(f"加载邻接矩阵: {path}")
            with open(path, 'rb') as f:
                adj_mx = pickle.load(f)
            
            # 处理不同格式的邻接矩阵
            if isinstance(adj_mx, tuple) and len(adj_mx) >= 3:
                # PEMS-BAY格式: (node_ids, distances, adjacency_matrix)
                self.node_ids, distances, adj_mx = adj_mx
                adj_matrix = adj_mx
            else:
                adj_matrix = adj_mx
            
            # 确保邻接矩阵形状正确
            if adj_matrix.shape != (self.num_nodes, self.num_nodes):
                logger.warning(f"邻接矩阵形状 {adj_matrix.shape} 与期望的 {(self.num_nodes, self.num_nodes)} 不匹配")
                # 尝试调整大小
                adj_matrix = self._resize_adj_matrix(adj_matrix, self.num_nodes)
            
            self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            logger.info(f"邻接矩阵加载完成，形状: {self.adj_matrix.shape}")
            return self.adj_matrix
            
        except Exception as e:
            logger.error(f"加载邻接矩阵失败: {e}")
            # 如果加载失败，使用默认方法构建邻接矩阵
            return self.build_adj_matrix()
    
    def _resize_adj_matrix(self, adj_matrix: np.ndarray, target_size: int) -> np.ndarray:
        """
        调整邻接矩阵大小
        
        Args:
            adj_matrix: 原始邻接矩阵
            target_size: 目标大小
            
        Returns:
            调整大小后的邻接矩阵
        """
        current_size = adj_matrix.shape[0]
        new_adj = np.zeros((target_size, target_size))
        
        # 复制现有数据
        min_size = min(current_size, target_size)
        new_adj[:min_size, :min_size] = adj_matrix[:min_size, :min_size]
        
        # 对新增的部分添加自环
        for i in range(min_size, target_size):
            new_adj[i, i] = 1.0
        
        return new_adj
    
    def build_adj_matrix(self, threshold: float = 0.1) -> torch.Tensor:
        """
        构建邻接矩阵
        
        如果有元数据，使用地理距离构建邻接矩阵；否则使用默认的邻接矩阵
        
        Args:
            threshold: 距离阈值，小于此阈值的节点视为相连
            
        Returns:
            邻接矩阵 [num_nodes, num_nodes]
        """
        if self.adj_matrix is not None:
            return self.adj_matrix
        
        try:
            if hasattr(self, 'metadata') and self.metadata is not None and 'Latitude' in self.metadata.columns and 'Longitude' in self.metadata.columns:
                # 使用地理距离构建邻接矩阵
                num_nodes = min(len(self.metadata), self.num_nodes)
                adj = np.zeros((num_nodes, num_nodes))
                
                # 计算节点间距离
                for i in range(num_nodes):
                    lat1, lon1 = self.metadata.iloc[i][['Latitude', 'Longitude']]
                    for j in range(num_nodes):
                        if i != j:
                            lat2, lon2 = self.metadata.iloc[j][['Latitude', 'Longitude']]
                            # 简单的欧几里得距离计算
                            distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                            adj[i][j] = 1.0 if distance < threshold else 0.0
            else:
                # 创建默认的邻接矩阵（简单的k近邻）
                num_nodes = self.num_nodes
                adj = np.zeros((num_nodes, num_nodes))
                
                # 创建一个环形拓扑结构
                for i in range(num_nodes):
                    adj[i][i] = 1.0  # 自环
                    adj[i][(i+1) % num_nodes] = 1.0  # 前向连接
                    adj[i][(i-1) % num_nodes] = 1.0  # 后向连接
                    
                    # 连接更远的节点
                    if i + 2 < num_nodes:
                        adj[i][(i+2) % num_nodes] = 0.5
                    if i - 2 >= 0:
                        adj[i][(i-2) % num_nodes] = 0.5
            
            # 归一化邻接矩阵
            degree = np.sum(adj, axis=1)
            degree_inv_sqrt = np.power(degree, -0.5, where=degree != 0)
            degree_matrix = np.diag(degree_inv_sqrt)
            adj = degree_matrix @ adj @ degree_matrix
            
            self.adj_matrix = torch.tensor(adj, dtype=torch.float32)
            logger.info(f"邻接矩阵构建完成，形状: {self.adj_matrix.shape}")
            return self.adj_matrix
            
        except Exception as e:
            logger.error(f"构建邻接矩阵失败: {e}")
            raise
    
    def create_sequences(self, sequence_length: int = 12, prediction_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建时空序列数据
        
        Args:
            sequence_length: 输入序列长度
            prediction_steps: 预测步数
            
        Returns:
            X: 输入序列 [num_samples, sequence_length, num_nodes, 1]
            y: 目标序列 [num_samples, prediction_steps, num_nodes, 1]
        """
        try:
            # 提取数值数据（不包括时间戳列）
            traffic_data = self.raw_data.drop('timestamp', axis=1).values
            
            # 确保数据形状正确
            if traffic_data.shape[1] != self.num_nodes:
                logger.warning(f"数据节点数 {traffic_data.shape[1]} 与期望的 {self.num_nodes} 不匹配")
                # 如果数据节点数多于期望，截断
                if traffic_data.shape[1] > self.num_nodes:
                    traffic_data = traffic_data[:, :self.num_nodes]
                # 如果数据节点数少于期望，填充
                elif traffic_data.shape[1] < self.num_nodes:
                    pad_width = ((0, 0), (0, self.num_nodes - traffic_data.shape[1]))
                    traffic_data = np.pad(traffic_data, pad_width, mode='constant', constant_values=0)
            
            # 转置数据以适应 [num_nodes, time_steps] 格式
            traffic_data = traffic_data.T  # [num_nodes, time_steps]
            
            # 处理缺失值
            traffic_data = np.nan_to_num(traffic_data)
            
            # 数据标准化
            num_nodes, time_steps = traffic_data.shape
            traffic_data_flat = traffic_data.reshape(-1, 1)  # 展平以进行标准化
            traffic_data_flat_scaled = self.scaler.fit_transform(traffic_data_flat)
            traffic_data_scaled = traffic_data_flat_scaled.reshape(num_nodes, time_steps)
            
            # 创建序列
            X, y = [], []
            total_sequence_length = sequence_length + prediction_steps
            
            for i in range(time_steps - total_sequence_length + 1):
                # 输入序列
                x_seq = traffic_data_scaled[:, i:i+sequence_length]
                X.append(x_seq)
                
                # 目标序列
                y_seq = traffic_data_scaled[:, i+sequence_length:i+total_sequence_length]
                y.append(y_seq)
            
            # 转换为张量并调整形状
            X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
            y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
            
            # 调整为 [num_samples, sequence_length, num_nodes, 1] 格式
            X_tensor = X_tensor.permute(0, 2, 1).unsqueeze(-1)
            y_tensor = y_tensor.permute(0, 2, 1).unsqueeze(-1)
            
            logger.info(f"序列创建完成，X形状: {X_tensor.shape}, y形状: {y_tensor.shape}")
            return X_tensor, y_tensor
            
        except Exception as e:
            logger.error(f"创建序列失败: {e}")
            raise
    
    def get_data_loaders(self, batch_size: int = 32, sequence_length: int = None, 
                        prediction_steps: int = None, shuffle: bool = True, 
                        train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        获取训练、验证和测试数据加载器
        
        Args:
            batch_size: 批次大小
            sequence_length: 输入序列长度
            prediction_steps: 预测步数
            shuffle: 是否打乱数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            train_loader, val_loader, test_loader
        """
        try:
            # 使用默认值或传入的值
            sequence_length = sequence_length or self.default_sequence_length
            prediction_steps = prediction_steps or self.default_prediction_steps
            
            # 创建序列数据
            X, y = self.create_sequences(sequence_length, prediction_steps)
            
            # 创建数据集
            dataset = TensorDataset(X, y)
            
            # 计算分割大小
            total_size = len(dataset)
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * val_ratio)
            test_size = total_size - train_size - val_size
            
            # 分割数据集
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            logger.info(f"数据加载器创建完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.error(f"创建数据加载器失败: {e}")
            raise
    
    def get_standard_scaler(self):
        """
        获取数据标准化器
        
        Returns:
            用于数据标准化的StandardScaler对象
        """
        return self.scaler
    
    def inverse_transform(self, data):
        """
        将标准化后的数据转换回原始尺度
        
        Args:
            data: 标准化后的数据
            
        Returns:
            原始尺度的数据
        """
        try:
            # 确保数据是numpy数组
            if isinstance(data, torch.Tensor):
                data_np = data.cpu().numpy()
            else:
                data_np = data
            
            # 保存原始形状
            original_shape = data_np.shape
            
            # 展平数据以应用逆变换
            data_flat = data_np.reshape(-1, 1)
            data_inv = self.scaler.inverse_transform(data_flat)
            
            # 恢复原始形状
            return data_inv.reshape(original_shape)
            
        except Exception as e:
            logger.error(f"逆变换失败: {e}")
            raise

def main():
    """主函数，用于测试数据加载器"""
    try:
        # 初始化数据加载器
        data_dir = "d:\gcn-lstm\data\real-data"
        loader = RealTrafficDataLoader(data_dir, "METR-LA")
        
        # 构建邻接矩阵
        adj_matrix = loader.build_adj_matrix()
        print(f"邻接矩阵形状: {adj_matrix.shape}")
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = loader.get_data_loaders(batch_size=32)
        print(f"训练集批次: {len(train_loader)}")
        print(f"验证集批次: {len(val_loader)}")
        print(f"测试集批次: {len(test_loader)}")
        
        # 检查数据批次
        for X, y in train_loader:
            print(f"输入形状: {X.shape}")
            print(f"目标形状: {y.shape}")
            break
            
    except Exception as e:
        logger.error(f"测试失败: {e}")

if __name__ == "__main__":
    main()