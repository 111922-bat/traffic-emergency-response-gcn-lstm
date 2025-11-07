#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基线模型标准化训练脚本

此脚本提供了一个统一的框架，用于训练和评估各种基线模型，
确保所有模型在相同的条件下进行公平比较。支持模型包括：
- GCN (图卷积网络)
- LSTM (长短期记忆网络)
- STGCN (时空图卷积网络)
- T-GCN (时序图卷积网络)
- GCNLSTMHybrid (GCN-LSTM混合模型)
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_integration.preprocess import TrafficDataPreprocessor
from data_integration.real_traffic_data_loader import RealTrafficDataLoader
from config.config_loader import ExperimentConfig
from models.gcn_lstm_hybrid import GCNLSTMHybrid, ModelConfig
from evaluation.metrics import compute_metrics

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """
    所有模型的基类，定义统一接口
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x, adj=None):
        raise NotImplementedError
    
    def get_model_config(self):
        """
        获取模型配置信息
        """
        raise NotImplementedError


class GCNModel(BaseModel):
    """
    图卷积网络模型
    """
    def __init__(self, in_channels, hidden_dims, out_channels, num_nodes, dropout=0.2):
        super().__init__()
        self.name = "GCN"
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        
        # 构建GCN层
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(nn.Linear(in_channels, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.gcn_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        self.gcn_layers.append(nn.Linear(hidden_dims[-1], out_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x, adj=None):
        # x shape: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = x.shape
        
        # 将序列维度和节点维度合并
        x = x.reshape(batch_size, seq_len, -1)
        
        # 使用最后一个时间步作为输入
        x = x[:, -1, :].reshape(batch_size, num_nodes, in_channels)
        
        # GCN前向传播
        for i, layer in enumerate(self.gcn_layers[:-1]):
            if adj is not None:
                # 应用图卷积
                x = torch.matmul(adj, x)
            
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # 输出层
        x = self.gcn_layers[-1](x)
        
        # 返回形状: [batch_size, num_nodes, out_channels]
        return x
    
    def get_model_config(self):
        return {
            "name": self.name,
            "in_channels": self.in_channels,
            "hidden_dims": self.hidden_dims,
            "out_channels": self.out_channels,
            "num_nodes": self.num_nodes
        }


class LSTMModel(BaseModel):
    """
    长短期记忆网络模型
    """
    def __init__(self, in_channels, hidden_dims, out_channels, num_nodes, num_layers=2, dropout=0.2):
        super().__init__()
        self.name = "LSTM"
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=in_channels * num_nodes,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims, out_channels * num_nodes)
    
    def forward(self, x, adj=None):
        # x shape: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = x.shape
        
        # 将节点维度和特征维度合并
        x = x.reshape(batch_size, seq_len, -1)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 只使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 输出层
        output = self.output_layer(last_output)
        
        # 重塑为需要的输出形状
        output = output.reshape(batch_size, num_nodes, self.out_channels)
        
        return output
    
    def get_model_config(self):
        return {
            "name": self.name,
            "in_channels": self.in_channels,
            "hidden_dims": self.hidden_dims,
            "out_channels": self.out_channels,
            "num_nodes": self.num_nodes,
            "num_layers": self.num_layers
        }


class STGCNModel(BaseModel):
    """
    时空图卷积网络模型
    """
    def __init__(self, in_channels, hidden_dims, out_channels, num_nodes, dropout=0.2):
        super().__init__()
        self.name = "STGCN"
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        
        # 空间卷积层
        self.spatial_conv = nn.Linear(in_channels, hidden_dims)
        
        # 时间卷积层
        self.temporal_conv = nn.Conv1d(hidden_dims, hidden_dims, kernel_size=3, padding=1)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x, adj=None):
        # x shape: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = x.shape
        
        # 时间维度处理
        outputs = []
        
        for t in range(seq_len):
            # 获取当前时间步的特征
            x_t = x[:, t, :, :]
            
            # 空间卷积
            if adj is not None:
                x_t = torch.matmul(adj, x_t)
            
            x_t = self.spatial_conv(x_t)
            x_t = self.relu(x_t)
            outputs.append(x_t)
        
        # 堆叠时间维度
        x = torch.stack(outputs, dim=1)
        
        # 交换维度以适应1D卷积
        x = x.transpose(2, 3).reshape(batch_size * num_nodes, self.hidden_dims, seq_len)
        
        # 时间卷积
        x = self.temporal_conv(x)
        x = self.relu(x)
        
        # 获取最后一个时间步
        x = x[:, :, -1]
        
        # 重塑并输出
        x = x.reshape(batch_size, num_nodes, self.hidden_dims)
        x = self.output_layer(x)
        
        return x
    
    def get_model_config(self):
        return {
            "name": self.name,
            "in_channels": self.in_channels,
            "hidden_dims": self.hidden_dims,
            "out_channels": self.out_channels,
            "num_nodes": self.num_nodes
        }


class TGCNModel(BaseModel):
    """
    时序图卷积网络模型 (T-GCN)
    """
    def __init__(self, in_channels, hidden_dims, out_channels, num_nodes, num_layers=1, dropout=0.2):
        super().__init__()
        self.name = "T-GCN"
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        
        # GCN层
        self.gcn = nn.Linear(in_channels, hidden_dims)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dims,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x, adj=None):
        # x shape: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = x.shape
        
        # 对每个时间步应用GCN
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            
            # 应用图卷积
            if adj is not None:
                x_t = torch.matmul(adj, x_t)
            
            x_t = self.gcn(x_t)
            x_t = self.relu(x_t)
            gcn_outputs.append(x_t)
        
        # 堆叠并重塑为LSTM输入格式
        x = torch.stack(gcn_outputs, dim=1)
        x = x.reshape(batch_size, seq_len, num_nodes, -1)
        
        # 交换维度以将节点特征和时间特征组合
        lstm_input = []
        for b in range(batch_size):
            node_series = []
            for n in range(num_nodes):
                # 提取每个节点的时间序列
                node_series.append(x[b, :, n, :])
            lstm_input.append(torch.stack(node_series, dim=0))
        
        lstm_input = torch.stack(lstm_input, dim=0)
        
        # LSTM前向传播（每个节点单独处理）
        outputs = []
        for n in range(num_nodes):
            lstm_out, _ = self.lstm(lstm_input[:, n, :, :])
            outputs.append(lstm_out[:, -1, :])  # 取最后一个时间步
        
        # 合并结果
        x = torch.stack(outputs, dim=1)
        
        # 输出层
        x = self.output_layer(x)
        
        return x
    
    def get_model_config(self):
        return {
            "name": self.name,
            "in_channels": self.in_channels,
            "hidden_dims": self.hidden_dims,
            "out_channels": self.out_channels,
            "num_nodes": self.num_nodes
        }


class ModelTrainer:
    """
    模型训练器类，提供标准化的训练流程
    """
    def __init__(self, config: ExperimentConfig, model_type: str):
        """
        初始化模型训练器
        
        Args:
            config: 实验配置对象
            model_type: 模型类型
        """
        self.config = config
        self.model_type = model_type.lower()
        
        # 确保输出目录存在
        self.output_dir = os.path.join(
            config.output.save_dir, 
            model_type,
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(self.output_dir, f"{model_type}_training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # 创建TensorBoard写入器
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard"))
        
        # 设置设备
        self.device = torch.device(config.environment.device if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler = None
        
        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_model_path = os.path.join(self.output_dir, "best_model.pth")
        self.patience_counter = 0
    
    def load_data(self):
        """
        加载数据集
        """
        logger.info("开始加载数据...")
        
        # 创建数据预处理器
        preprocessor = TrafficDataPreprocessor(
            data_dir=self.config.dataset.data_dir,
            dataset_name=self.config.dataset.name,
            seq_len=self.config.dataset.seq_len,
            pred_len=self.config.dataset.pred_len,
            train_ratio=self.config.dataset.train_ratio,
            val_ratio=self.config.dataset.val_ratio,
            test_ratio=self.config.dataset.test_ratio,
            random_seed=self.config.random_seed
        )
        
        # 处理数据
        preprocessor.preprocess()
        
        # 创建数据加载器
        data_loader = RealTrafficDataLoader(
            data_dir=self.config.dataset.data_dir,
            dataset_name=self.config.dataset.name,
            batch_size=self.config.training.batch_size,
            seq_len=self.config.dataset.seq_len,
            pred_len=self.config.dataset.pred_len,
            num_workers=self.config.environment.num_workers,
            pin_memory=self.config.environment.pin_memory
        )
        
        self.train_loader, self.val_loader, self.test_loader = data_loader.get_loaders()
        self.scaler = data_loader.get_scaler()
        
        # 获取图结构（如果需要）
        self.adj_matrix = data_loader.get_adjacency_matrix()
        if self.adj_matrix is not None:
            self.adj_matrix = torch.tensor(self.adj_matrix, dtype=torch.float32).to(self.device)
        
        # 记录数据信息
        logger.info(f"数据集: {self.config.dataset.name}")
        logger.info(f"训练集批次: {len(self.train_loader)}")
        logger.info(f"验证集批次: {len(self.val_loader)}")
        logger.info(f"测试集批次: {len(self.test_loader)}")
        
        # 获取样本数据以确定模型参数
        sample_batch = next(iter(self.train_loader))
        x, y = sample_batch
        self.num_nodes = x.shape[2]
        self.in_channels = x.shape[3]
        self.out_channels = y.shape[3]
        
        logger.info(f"输入特征维度: {self.in_channels}")
        logger.info(f"输出特征维度: {self.out_channels}")
        logger.info(f"节点数量: {self.num_nodes}")
    
    def create_model(self):
        """
        创建指定类型的模型
        """
        logger.info(f"创建 {self.model_type} 模型...")
        
        # 从配置中获取模型参数
        model_config = self.config.models_config.get(self.model_type)
        
        if self.model_type == "gcn":
            # 创建GCN模型
            self.model = GCNModel(
                in_channels=self.in_channels,
                hidden_dims=[model_config.get("hidden_dim", 64)] * model_config.get("gcn_layers", 2),
                out_channels=self.out_channels,
                num_nodes=self.num_nodes,
                dropout=model_config.get("dropout", 0.2)
            )
        
        elif self.model_type == "lstm":
            # 创建LSTM模型
            self.model = LSTMModel(
                in_channels=self.in_channels,
                hidden_dims=model_config.get("hidden_dim", 64),
                out_channels=self.out_channels,
                num_nodes=self.num_nodes,
                num_layers=model_config.get("lstm_layers", 2),
                dropout=model_config.get("dropout", 0.2)
            )
        
        elif self.model_type == "stgcn":
            # 创建STGCN模型
            self.model = STGCNModel(
                in_channels=self.in_channels,
                hidden_dims=model_config.get("hidden_dim", 64),
                out_channels=self.out_channels,
                num_nodes=self.num_nodes,
                dropout=model_config.get("dropout", 0.2)
            )
        
        elif self.model_type == "t-gcn":
            # 创建T-GCN模型
            self.model = TGCNModel(
                in_channels=self.in_channels,
                hidden_dims=model_config.get("hidden_dim", 64),
                out_channels=self.out_channels,
                num_nodes=self.num_nodes,
                num_layers=model_config.get("lstm_layers", 1),
                dropout=model_config.get("dropout", 0.2)
            )
        
        elif self.model_type == "gcnlstmhybrid":
            # 使用现有的GCNLSTMHybrid模型
            hybrid_config = ModelConfig(
                num_nodes=self.num_nodes,
                input_dim=self.in_channels,
                output_dim=self.out_channels,
                gcn_layers=model_config.get("gcn_layers", 2),
                lstm_layers=model_config.get("lstm_layers", 2),
                gcn_hidden_dim=model_config.get("gcn_hidden_dim", 64),
                lstm_hidden_dim=model_config.get("lstm_hidden_dim", 64),
                dropout=model_config.get("dropout", 0.2),
                fusion_strategy=model_config.get("fusion_strategy", "attention"),
                use_spatial_attention=model_config.get("use_spatial_attention", True)
            )
            self.model = GCNLSTMHybrid(hybrid_config)
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 将模型移至设备
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        logger.info(f"模型结构: {self.model}")
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        # 保存模型配置
        model_info = self.model.get_model_config()
        with open(os.path.join(self.output_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    def setup_optimizer(self):
        """
        设置优化器和学习率调度器
        """
        # 选择优化器
        if self.config.training.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=1e-4
            )
        elif self.config.training.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif self.config.training.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.training.optimizer}")
        
        # 设置学习率调度器
        if self.config.training.lr_decay > 0:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_step,
                gamma=self.config.training.lr_decay
            )
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            # 将数据移至设备
            x = x.to(self.device)
            y = y.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(x, self.adj_matrix)
            
            # 计算损失
            loss = self.criterion(output, y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.training.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            
            # 记录训练进度
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch}/{self.config.training.epochs}], "
                           f"Batch [{batch_idx + 1}/{len(self.train_loader)}], "
                           f"Loss: {loss.item():.6f}")
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        
        # 记录到TensorBoard
        self.tb_writer.add_scalar("Loss/train", avg_loss, epoch)
        self.tb_writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], epoch)
        
        logger.info(f"Epoch [{epoch}/{self.config.training.epochs}], Train Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def evaluate(self, data_loader, split_name="val"):
        """
        在验证集或测试集上评估模型
        
        Args:
            data_loader: 数据加载器
            split_name: 数据集分割名称 ("val" 或 "test")
            
        Returns:
            Dict: 评估指标字典
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in data_loader:
                # 将数据移至设备
                x = x.to(self.device)
                y = y.to(self.device)
                
                # 前向传播
                output = self.model(x, self.adj_matrix)
                
                # 计算损失
                loss = self.criterion(output, y)
                total_loss += loss.item()
                
                # 保存预测和目标值
                all_predictions.append(output.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(data_loader)
        
        # 合并所有批次的预测和目标
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 反归一化（如果有scaler）
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
            targets = self.scaler.inverse_transform(targets)
        
        # 计算评估指标
        metrics = compute_metrics(predictions, targets, self.config.evaluation.metrics)
        metrics["loss"] = avg_loss
        
        # 记录到TensorBoard
        self.tb_writer.add_scalar(f"Loss/{split_name}", avg_loss, self.current_epoch)
        for metric_name, value in metrics.items():
            if metric_name != "loss":
                self.tb_writer.add_scalar(f"Metrics/{split_name}_{metric_name}", value, self.current_epoch)
        
        # 打印评估结果
        logger.info(f"{split_name.capitalize()} Evaluation: Loss={avg_loss:.6f}")
        for metric_name, value in metrics.items():
            if metric_name != "loss":
                logger.info(f"  {metric_name.upper()}: {value:.6f}")
        
        return metrics, predictions, targets
    
    def train(self):
        """
        训练模型
        """
        logger.info(f"开始训练 {self.model_type} 模型...")
        start_time = time.time()
        
        # 记录训练开始信息
        training_info = {
            "model_type": self.model_type,
            "start_time": datetime.datetime.now().isoformat(),
            "config": self.config.to_dict()
        }
        
        # 保存训练配置
        with open(os.path.join(self.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        # 训练历史记录
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "best_epoch": 0
        }
        
        # 开始训练循环
        for epoch in range(1, self.config.training.epochs + 1):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 评估模型
            val_metrics, _, _ = self.evaluate(self.val_loader, "val")
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 保存历史记录
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_metrics"].append(val_metrics)
            
            # 检查是否是最佳模型
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.best_val_loss,
                }, self.best_model_path)
                
                logger.info(f"保存最佳模型: {self.best_model_path} (Val Loss: {self.best_val_loss:.6f})")
            else:
                self.patience_counter += 1
                logger.info(f"早停计数器: {self.patience_counter}/{self.config.training.patience}")
            
            # 早停检查
            if self.config.training.use_early_stopping and \
               self.patience_counter >= self.config.training.patience:
                logger.info(f"早停: 验证损失不再改善")
                break
        
        # 训练结束
        end_time = time.time()
        training_time = end_time - start_time
        
        # 记录训练结束信息
        history["best_epoch"] = self.best_epoch
        history["training_time"] = training_time
        history["end_time"] = datetime.datetime.now().isoformat()
        
        # 保存训练历史
        with open(os.path.join(self.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练完成！")
        logger.info(f"总训练时间: {training_time:.2f} 秒")
        logger.info(f"最佳模型 epoch: {self.best_epoch}")
        logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
        
        return history
    
    def test(self):
        """
        在测试集上评估最佳模型
        """
        logger.info("开始测试最佳模型...")
        
        # 加载最佳模型
        checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已加载最佳模型 (epoch {checkpoint['epoch']})")
        
        # 在测试集上评估
        test_metrics, predictions, targets = self.evaluate(self.test_loader, "test")
        
        # 保存测试结果
        test_results = {
            "model_type": self.model_type,
            "best_epoch": checkpoint['epoch'],
            "test_metrics": test_metrics,
            "test_time": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        # 保存预测结果样本
        if self.config.evaluation.save_predictions:
            np.savez_compressed(
                os.path.join(self.output_dir, "predictions.npz"),
                predictions=predictions[:100],  # 只保存前100个样本
                targets=targets[:100]
            )
        
        logger.info("测试完成！")
        logger.info(f"测试指标:")
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name.upper()}: {value:.6f}")
        
        return test_results


def setup_experiment(config_file: str, model_type: str, run_id: int = 0):
    """
    设置实验环境和配置
    
    Args:
        config_file: 配置文件路径
        model_type: 模型类型
        run_id: 运行ID（用于多次运行）
    """
    # 加载配置
    config = ExperimentConfig(config_file)
    
    # 设置随机种子确保可复现性
    seed = config.random_seed + run_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置cuDNN确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return config


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="基线模型训练器")
    parser.add_argument("--config", type=str, default="../configs/experiment_config.yaml",
                        help="实验配置文件路径")
    parser.add_argument("--model", type=str, required=True,
                        choices=["GCN", "LSTM", "STGCN", "T-GCN", "GCNLSTMHybrid"],
                        help="要训练的模型类型")
    parser.add_argument("--runs", type=int, default=1,
                        help="运行次数，用于计算平均值和标准差")
    parser.add_argument("--gpu", type=int, default=0,
                        help="使用的GPU索引，如果为-1则使用CPU")
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_arguments()
    
    # 保存所有运行的结果
    all_runs_results = []
    
    # 多次运行模型
    for run_id in range(args.runs):
        logger.info(f"\n========== 运行 {run_id + 1}/{args.runs} ==========")
        
        # 设置实验
        config = setup_experiment(args.config, args.model, run_id)
        
        # 如果指定了GPU索引
        if args.gpu >= 0 and torch.cuda.is_available():
            config.environment.device = f"cuda:{args.gpu}"
        else:
            config.environment.device = "cpu"
        
        # 创建训练器
        trainer = ModelTrainer(config, args.model)
        
        # 加载数据
        trainer.load_data()
        
        # 创建模型
        trainer.create_model()
        
        # 设置优化器
        trainer.setup_optimizer()
        
        # 训练模型
        training_history = trainer.train()
        
        # 测试模型
        test_results = trainer.test()
        
        # 保存运行结果
        run_result = {
            "run_id": run_id,
            "model_type": args.model,
            "output_dir": trainer.output_dir,
            "best_epoch": training_history["best_epoch"],
            "best_val_loss": training_history["val_loss"][training_history["best_epoch"] - 1],
            "test_metrics": test_results["test_metrics"],
            "training_time": training_history["training_time"]
        }
        
        all_runs_results.append(run_result)
    
    # 如果运行了多次，计算统计结果
    if args.runs > 1:
        logger.info(f"\n========== {args.runs}次运行结果统计 ==========")
        
        # 计算每个指标的平均值和标准差
        metric_stats = {}
        metrics = all_runs_results[0]["test_metrics"].keys()
        
        for metric in metrics:
            values = [run["test_metrics"][metric] for run in all_runs_results]
            metric_stats[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }
            
            logger.info(f"{metric.upper()}: {np.mean(values):.6f} ± {np.std(values):.6f}")
        
        # 计算训练时间统计
        training_times = [run["training_time"] for run in all_runs_results]
        time_stats = {
            "mean": np.mean(training_times),
            "std": np.std(training_times)
        }
        logger.info(f"平均训练时间: {np.mean(training_times):.2f} ± {np.std(training_times):.2f} 秒")
        
        # 保存统计结果
        stats_result = {
            "model_type": args.model,
            "num_runs": args.runs,
            "metric_stats": metric_stats,
            "time_stats": time_stats,
            "all_runs": all_runs_results,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # 创建结果汇总目录
        summary_dir = os.path.join(config.output.save_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 保存统计结果
        with open(os.path.join(summary_dir, f"{args.model.lower()}_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"统计结果已保存到: {os.path.join(summary_dir, f'{args.model.lower()}_stats.json')}")


if __name__ == "__main__":
    main()