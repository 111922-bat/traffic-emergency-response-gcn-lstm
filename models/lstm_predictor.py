"""
LSTM时序预测模型
支持多步预测、注意力机制、批量处理等功能，专门针对交通流时序数据优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class AttentionMechanism(nn.Module):
    """注意力机制模块"""
    
    def __init__(self, hidden_size: int):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.v.data.normal_(0, 0.1)
        
    def forward(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            encoder_outputs: 编码器输出 [batch_size, seq_len, hidden_size]
            
        Returns:
            context: 上下文向量 [batch_size, hidden_size]
            attn_weights: 注意力权重 [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_size = encoder_outputs.size()
        
        # 计算注意力分数
        energy = torch.tanh(self.attn(encoder_outputs))  # [batch_size, seq_len, hidden_size]
        energy = energy.view(batch_size, seq_len, hidden_size)
        
        # 计算注意力权重
        v = self.v.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        v = v.expand(batch_size, 1, hidden_size)  # [batch_size, 1, hidden_size]
        
        # 计算注意力分数
        energy = energy.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        attn_weights = torch.bmm(v, energy)  # [batch_size, 1, seq_len]
        attn_weights = attn_weights.squeeze(1)  # [batch_size, seq_len]
        
        # Softmax归一化
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch_size, seq_len]
        
        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        return context, attn_weights


class LSTMPredictor(nn.Module):
    """LSTM时序预测模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 num_steps: int = 1,
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 use_batch_norm: bool = True):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            num_steps: 预测步数
            dropout: Dropout比例
            use_attention: 是否使用注意力机制
            use_batch_norm: 是否使用批归一化
        """
        super(LSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_steps = num_steps
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 批归一化
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # 注意力机制
        if use_attention:
            self.attention = AttentionMechanism(hidden_size)
        
        # 全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size * num_steps)
        self.relu = nn.ReLU()
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            
        Returns:
            output: 预测结果 [batch_size, num_steps * output_size]
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, seq_len, hidden_size]
        
        # 注意力机制
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
            # 使用最后一个时间步的输出和注意力上下文
            output = lstm_out[:, -1, :] + context
        else:
            # 使用最后一个时间步的输出
            output = lstm_out[:, -1, :]
        
        # 批归一化
        if self.use_batch_norm:
            output = self.batch_norm(output)
        
        # 全连接层
        output = self.dropout(output)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        
        # 重塑输出为 [batch_size, num_steps, output_size]
        output = output.view(output.size(0), self.num_steps, self.output_size)
        
        return output


class Seq2SeqLSTMPredictor(nn.Module):
    """序列到序列LSTM预测模型"""
    
    def __init__(self,
                 input_size: int,
                 encoder_hidden_size: int = 128,
                 decoder_hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 use_attention: bool = True):
        """
        初始化Seq2Seq LSTM模型
        
        Args:
            input_size: 输入特征维度
            encoder_hidden_size: 编码器隐藏层大小
            decoder_hidden_size: 解码器隐藏层大小
            num_layers: 层数
            output_size: 输出维度
            dropout: Dropout比例
            use_attention: 是否使用注意力机制
        """
        super(Seq2SeqLSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_attention = use_attention
        
        # 编码器
        self.encoder = nn.LSTM(input_size, encoder_hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 解码器
        self.decoder = nn.LSTM(input_size, decoder_hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 注意力机制
        if use_attention:
            self.attention = AttentionMechanism(encoder_hidden_size)
        
        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_hidden_size, output_size)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.encoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        for name, param in self.decoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_seq_len, input_size]
            tgt: 目标序列 [batch_size, tgt_seq_len, input_size] (训练时需要)
            
        Returns:
            output: 预测序列 [batch_size, tgt_seq_len, output_size]
        """
        batch_size, src_seq_len, _ = src.size()
        
        # 编码
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # 解码
        if self.training and tgt is not None:
            # 训练模式：使用真实目标序列
            tgt_seq_len = tgt.size(1)
            decoder_outputs = []
            
            # 初始化解码器隐藏状态
            decoder_hidden = hidden
            decoder_cell = cell
            
            for t in range(tgt_seq_len):
                # 解码器输入
                decoder_input = tgt[:, t:t+1, :]  # [batch_size, 1, input_size]
                
                # 解码器前向传播
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                    decoder_input, (decoder_hidden, decoder_cell))
                
                # 注意力机制
                if self.use_attention:
                    context, _ = self.attention(encoder_outputs)
                    decoder_output = decoder_output.squeeze(1) + context
                    decoder_output = decoder_output.unsqueeze(1)
                
                # 输出层
                output = self.fc(decoder_output.squeeze(1))  # [batch_size, output_size]
                decoder_outputs.append(output)
            
            output = torch.stack(decoder_outputs, dim=1)  # [batch_size, tgt_seq_len, output_size]
        else:
            # 推理模式：自回归生成
            tgt_seq_len = src_seq_len  # 假设预测序列长度与输入相同
            decoder_outputs = []
            
            # 初始化解码器隐藏状态
            decoder_hidden = hidden
            decoder_cell = cell
            
            # 使用最后一个输入作为解码器的第一个输入
            decoder_input = src[:, -1:, :]
            
            for t in range(tgt_seq_len):
                # 解码器前向传播
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                    decoder_input, (decoder_hidden, decoder_cell))
                
                # 注意力机制
                if self.use_attention:
                    context, _ = self.attention(encoder_outputs)
                    decoder_output = decoder_output.squeeze(1) + context
                    decoder_output = decoder_output.unsqueeze(1)
                
                # 输出层
                output = self.fc(decoder_output.squeeze(1))  # [batch_size, output_size]
                decoder_outputs.append(output)
                
                # 使用当前输出作为下一个时间步的输入
                # 创建一个与输入特征维度匹配的占位符
                decoder_input = torch.zeros_like(src[:, 0:1, :])  # [batch_size, 1, input_size]
                decoder_input[:, :, -1] = output  # 将预测值放在最后一维
            
            output = torch.stack(decoder_outputs, dim=1)  # [batch_size, tgt_seq_len, output_size]
        
        return output


class LSTMDataProcessor:
    """LSTM数据预处理器"""
    
    def __init__(self, 
                 sequence_length: int = 24,
                 prediction_steps: int = 1,
                 feature_columns: Optional[List[str]] = None,
                 target_column: str = 'flow',
                 scaler_type: str = 'standard'):
        """
        初始化数据处理器
        
        Args:
            sequence_length: 序列长度
            prediction_steps: 预测步数
            feature_columns: 特征列名
            target_column: 目标列名
            scaler_type: 标准化器类型 ('standard' 或 'minmax')
        """
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaler_type = scaler_type
        
        # 初始化标准化器
        if scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
            
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'LSTMDataProcessor':
        """
        拟合数据处理器
        
        Args:
            data: 输入数据
            
        Returns:
            self: 返回自身实例
        """
        if self.feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != self.target_column]
        
        # 拟合特征标准化器
        self.feature_scaler.fit(data[self.feature_columns])
        
        # 拟合目标标准化器
        self.target_scaler.fit(data[[self.target_column]])
        
        self.is_fitted = True
        return self
        
    def transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换数据
        
        Args:
            data: 输入数据
            
        Returns:
            X: 特征序列 [num_samples, sequence_length, num_features]
            y: 目标序列 [num_samples, prediction_steps]
        """
        if not self.is_fitted:
            raise ValueError("数据处理器尚未拟合，请先调用fit方法")
        
        # 标准化数据
        features_scaled = self.feature_scaler.transform(data[self.feature_columns])
        targets_scaled = self.target_scaler.transform(data[[self.target_column]])
        
        # 创建序列数据
        X, y = self._create_sequences(features_scaled, targets_scaled)
        
        return X, y
        
    def fit_transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合并转换数据
        
        Args:
            data: 输入数据
            
        Returns:
            X: 特征序列
            y: 目标序列
        """
        return self.fit(data).transform(data)
        
    def _create_sequences(self, 
                         features: np.ndarray, 
                         targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据
        
        Args:
            features: 标准化后的特征
            targets: 标准化后的目标
            
        Returns:
            X: 特征序列
            y: 目标序列
        """
        num_samples = len(features)
        num_features = features.shape[1]
        
        # 计算有效样本数
        effective_samples = num_samples - self.sequence_length - self.prediction_steps + 1
        if effective_samples <= 0:
            raise ValueError(f"数据量不足，需要至少 {self.sequence_length + self.prediction_steps} 个样本")
        
        # 初始化序列数组
        X = np.zeros((effective_samples, self.sequence_length, num_features))
        y = np.zeros((effective_samples, self.prediction_steps))
        
        # 创建序列
        for i in range(effective_samples):
            # 特征序列
            X[i] = features[i:i + self.sequence_length]
            
            # 目标序列
            y[i] = targets[i + self.sequence_length:i + self.sequence_length + self.prediction_steps].flatten()
        
        return X, y
        
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        反向转换目标值
        
        Args:
            y_scaled: 标准化后的目标值
            
        Returns:
            y_original: 原始目标值
        """
        if not self.is_fitted:
            raise ValueError("数据处理器尚未拟合")
        
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(y_scaled)


class LSTMTrainer:
    """LSTM模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 device: str = 'auto'):
        """
        初始化训练器
        
        Args:
            model: LSTM模型
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 设备 ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                             mode='min', 
                                                             factor=0.5, 
                                                             patience=10)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            
            # 计算损失
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            avg_loss: 平均损失
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_X)
                
                # 计算损失
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader = None,
              num_epochs: int = 100,
              early_stopping_patience: int = 20,
              save_best_model: bool = True,
              save_path: str = 'best_model.pth') -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_best_model: 是否保存最佳模型
            save_path: 模型保存路径
            
        Returns:
            training_history: 训练历史
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始在设备 {self.device} 上训练模型...")
        print(f"训练样本数: {len(train_loader.dataset)}")
        if val_loader:
            print(f"验证样本数: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = None
            if val_loader:
                val_loss = self.validate_epoch(val_loader)
                self.scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if save_best_model:
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': epoch,
                            'loss': val_loss,
                        }, save_path)
                else:
                    patience_counter += 1
                
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")
                
                # 早停
                if patience_counter >= early_stopping_patience:
                    print(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
            else:
                self.scheduler.step(train_loss)
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}")
        
        training_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss if val_loader else None,
            'final_epoch': epoch + 1
        }
        
        return training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入数据 [batch_size, seq_len, features]
            
        Returns:
            predictions: 预测结果
        """
        self.model.eval()
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def save_model(self, save_path: str, processor: LSTMDataProcessor = None):
        """
        保存模型
        
        Args:
            save_path: 保存路径
            processor: 数据处理器
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_size': self.model.input_size if hasattr(self.model, 'input_size') else None,
                'hidden_size': self.model.hidden_size if hasattr(self.model, 'hidden_size') else None,
                'num_layers': self.model.num_layers if hasattr(self.model, 'num_layers') else None,
                'output_size': self.model.output_size if hasattr(self.model, 'output_size') else None,
                'num_steps': self.model.num_steps if hasattr(self.model, 'num_steps') else None,
            }
        }
        
        torch.save(checkpoint, save_path)
        
        # 保存数据处理器
        if processor:
            processor_path = save_path.replace('.pth', '_processor.pkl')
            with open(processor_path, 'wb') as f:
                pickle.dump(processor, f)
        
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str, processor_path: str = None) -> LSTMDataProcessor:
        """
        加载模型
        
        Args:
            load_path: 模型文件路径
            processor_path: 数据处理器文件路径
            
        Returns:
            processor: 数据处理器
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载数据处理器
        processor = None
        if processor_path and os.path.exists(processor_path):
            with open(processor_path, 'rb') as f:
                processor = pickle.load(f)
        
        print(f"模型已从 {load_path} 加载")
        return processor


class LSTMEvaluator:
    """LSTM模型评估器"""
    
    def __init__(self, processor: LSTMDataProcessor):
        """
        初始化评估器
        
        Args:
            processor: 数据处理器
        """
        self.processor = processor
        
    def evaluate(self, 
                model: nn.Module,
                X_test: np.ndarray,
                y_test: np.ndarray,
                device: str = 'auto') -> Dict:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            device: 设备
            
        Returns:
            metrics: 评估指标
        """
        # 设置设备
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        model.eval()
        model.to(device)
        
        # 转换为tensor并预测
        X_tensor = torch.FloatTensor(X_test).to(device)
        
        with torch.no_grad():
            predictions = model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        # 反向转换
        y_true_original = self.processor.inverse_transform_target(y_test)
        y_pred_original = self.processor.inverse_transform_target(predictions.flatten())
        
        # 计算评估指标
        mae = mean_absolute_error(y_true_original, y_pred_original)
        mse = mean_squared_error(y_true_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_original, y_pred_original)
        mape = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'predictions': y_pred_original,
            'true_values': y_true_original
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """打印评估指标"""
        print("\n=== 模型评估结果 ===")
        print(f"平均绝对误差 (MAE): {metrics['MAE']:.4f}")
        print(f"均方误差 (MSE): {metrics['MSE']:.4f}")
        print(f"均方根误差 (RMSE): {metrics['RMSE']:.4f}")
        print(f"决定系数 (R²): {metrics['R2']:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {metrics['MAPE']:.2f}%")


def create_data_loaders(X: np.ndarray, 
                       y: np.ndarray,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       batch_size: int = 32,
                       shuffle: bool = True) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    创建数据加载器
    
    Args:
        X: 特征数据
        y: 目标数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        batch_size: 批次大小
        shuffle: 是否打乱
        
    Returns:
        train_loader, val_loader, test_loader: 数据加载器
    """
    # 计算分割点
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 分割数据
    indices = np.arange(total_size)
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X[train_indices]), 
        torch.FloatTensor(y[train_indices])
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X[val_indices]), 
        torch.FloatTensor(y[val_indices])
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X[test_indices]), 
        torch.FloatTensor(y[test_indices])
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


# 示例用法和测试函数
def demo_lstm_predictor():
    """LSTM预测器演示"""
    print("=== LSTM时序预测模型演示 ===")
    
    # 生成模拟交通流数据
    np.random.seed(42)
    n_samples = 1000
    time_steps = 24  # 24小时数据
    
    # 生成模拟数据
    t = np.arange(n_samples)
    flow_data = []
    
    for i in range(n_samples):
        # 模拟交通流：日周期 + 噪声
        daily_pattern = 100 + 50 * np.sin(2 * np.pi * t[i] / 24) + 20 * np.cos(4 * np.pi * t[i] / 24)
        noise = np.random.normal(0, 10)
        flow = daily_pattern + noise
        flow_data.append(flow)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'hour': t % 24,
        'day_of_week': (t // 24) % 7,
        'flow': flow_data,
        'temperature': 20 + 10 * np.sin(2 * np.pi * t / (24 * 7)) + np.random.normal(0, 2, n_samples),
        'weather': np.random.randint(0, 4, n_samples)  # 天气状况
    })
    
    print(f"生成数据形状: {data.shape}")
    print(f"数据预览:\n{data.head()}")
    
    # 数据预处理
    processor = LSTMDataProcessor(
        sequence_length=12,  # 使用12小时历史数据
        prediction_steps=1,  # 预测下一步
        feature_columns=['hour', 'day_of_week', 'temperature', 'weather'],
        target_column='flow'
    )
    
    X, y = processor.fit_transform(data)
    print(f"处理后数据形状: X={X.shape}, y={y.shape}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32
    )
    
    # 创建模型
    model = LSTMPredictor(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        output_size=1,
        num_steps=1,
        dropout=0.2,
        use_attention=True,
        use_batch_norm=True
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    trainer = LSTMTrainer(model, learning_rate=0.001)
    history = trainer.train(
        train_loader, val_loader, 
        num_epochs=50, 
        early_stopping_patience=10,
        save_path='traffic_lstm_model.pth'
    )
    
    print(f"训练完成，最终验证损失: {history['best_val_loss']:.6f}")
    
    # 评估模型
    evaluator = LSTMEvaluator(processor)
    
    # 获取测试数据
    test_X, test_y = next(iter(test_loader))
    test_X = test_X.numpy()
    test_y = test_y.numpy()
    
    metrics = evaluator.evaluate(model, test_X, test_y)
    evaluator.print_metrics(metrics)
    
    # 保存模型和处理器
    trainer.save_model('traffic_lstm_model.pth', processor)
    
    # 演示预测
    print("\n=== 预测演示 ===")
    sample_input = test_X[:3]  # 取3个样本
    predictions = trainer.predict(sample_input)
    
    print("预测结果:")
    for i in range(len(sample_input)):
        true_val = processor.inverse_transform_target(test_y[i])[0]
        pred_val = processor.inverse_transform_target(predictions[i])[0]
        print(f"样本 {i+1}: 真实值={true_val:.2f}, 预测值={pred_val:.2f}")


if __name__ == "__main__":
    # 运行演示
    demo_lstm_predictor()