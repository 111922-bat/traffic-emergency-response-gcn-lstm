"""
数据预处理脚本

此脚本提供完整的数据预处理功能，包括：
- 数据加载和基本检查
- 缺失值处理
- 异常值检测和处理
- 数据标准化/归一化
- 特征工程
- 时间序列数据构建
- 训练/验证/测试集分割
- 数据版本控制支持

使用方法：
    python preprocess.py --dataset METR-LA --output_dir ../data/processed --random_seed 42
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from datetime import datetime
import json
from typing import Tuple, Dict, List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_preprocessor')

class TrafficDataPreprocessor:
    """交通数据预处理类"""
    
    def __init__(self, data_dir: str, dataset_name: str, random_seed: int = 42):
        """
        初始化数据预处理器
        
        Args:
            data_dir: 数据目录路径
            dataset_name: 数据集名称
            random_seed: 随机种子，用于确保结果可复现
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name.upper()
        self.random_seed = random_seed
        self.data_path = os.path.join(data_dir, f"{self.dataset_name}.csv")
        self.meta_path = os.path.join(data_dir, f"{self.dataset_name}-META.csv") if self.dataset_name == 'PEMS-BAY' else None
        self.adj_path = os.path.join(data_dir, "adj_mx_bay.pkl") if self.dataset_name == 'PEMS-BAY' else None
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # 数据集特定参数
        if self.dataset_name == 'METR-LA':
            self.num_nodes = 207
        elif self.dataset_name == 'PEMS-BAY':
            self.num_nodes = 325
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 数据存储
        self.raw_data = None
        self.processed_data = None
        self.scaler = None
        self.node_ids = None
        self.adj_matrix = None
        
        # 版本信息
        self.version = datetime.now().strftime("v%Y%m%d_%H%M")
        
        logger.info(f"初始化数据预处理器: {self.dataset_name}, 版本: {self.version}")
    
    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        Returns:
            加载的原始数据
        """
        logger.info(f"加载数据: {self.data_path}")
        try:
            # 读取CSV文件
            self.raw_data = pd.read_csv(self.data_path)
            
            # 处理时间戳列
            if 'Unnamed: 0' in self.raw_data.columns:
                self.raw_data.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
                self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
                self.node_ids = [col for col in self.raw_data.columns if col != 'timestamp']
            else:
                self.raw_data.columns = ['timestamp'] + [f'node_{i}' for i in range(1, len(self.raw_data.columns))]
                self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
                self.node_ids = [col for col in self.raw_data.columns if col != 'timestamp']
            
            logger.info(f"数据加载完成，形状: {self.raw_data.shape}")
            logger.info(f"包含 {len(self.node_ids)} 个节点，{len(self.raw_data)} 个时间点")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def check_data_quality(self) -> Dict[str, any]:
        """
        检查数据质量
        
        Returns:
            数据质量报告
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        
        logger.info("执行数据质量检查...")
        report = {
            'total_rows': len(self.raw_data),
            'total_columns': len(self.raw_data.columns),
            'missing_values': {},
            'negative_values': {},
            'statistics': {}
        }
        
        # 检查缺失值
        for col in self.node_ids:
            missing_count = self.raw_data[col].isna().sum()
            if missing_count > 0:
                report['missing_values'][col] = missing_count
        
        # 检查负值
        for col in self.node_ids:
            negative_count = (self.raw_data[col] < 0).sum()
            if negative_count > 0:
                report['negative_values'][col] = negative_count
        
        # 基本统计
        numeric_data = self.raw_data[self.node_ids]
        report['statistics'] = {
            'mean': numeric_data.mean().to_dict(),
            'std': numeric_data.std().to_dict(),
            'min': numeric_data.min().to_dict(),
            'max': numeric_data.max().to_dict()
        }
        
        logger.info(f"数据质量检查完成: 缺失值 {len(report['missing_values'])} 列, 负值 {len(report['negative_values'])} 列")
        return report
    
    def handle_missing_values(self, method: str = 'interpolate') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            method: 处理方法，支持 'interpolate', 'ffill', 'bfill', 'mean'
            
        Returns:
            处理后的数据
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        
        logger.info(f"处理缺失值，方法: {method}")
        self.processed_data = self.raw_data.copy()
        
        for col in self.node_ids:
            if self.processed_data[col].isna().sum() > 0:
                if method == 'interpolate':
                    self.processed_data[col] = self.processed_data[col].interpolate(method='linear')
                elif method == 'ffill':
                    self.processed_data[col] = self.processed_data[col].ffill()
                elif method == 'bfill':
                    self.processed_data[col] = self.processed_data[col].bfill()
                elif method == 'mean':
                    mean_value = self.processed_data[col].mean()
                    self.processed_data[col] = self.processed_data[col].fillna(mean_value)
                else:
                    raise ValueError(f"不支持的缺失值处理方法: {method}")
        
        # 确保没有剩余的缺失值
        for col in self.node_ids:
            self.processed_data[col] = self.processed_data[col].fillna(0)  # 兜底处理
        
        logger.info("缺失值处理完成")
        return self.processed_data
    
    def handle_outliers(self, method: str = 'clip', threshold: float = 3.0) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            method: 处理方法，支持 'clip', 'remove'
            threshold: 异常值检测阈值（标准差倍数）
            
        Returns:
            处理后的数据
        """
        if self.processed_data is None:
            raise ValueError("请先处理缺失值")
        
        logger.info(f"处理异常值，方法: {method}, 阈值: {threshold}")
        
        for col in self.node_ids:
            mean_val = self.processed_data[col].mean()
            std_val = self.processed_data[col].std()
            lower_bound = max(0, mean_val - threshold * std_val)  # 速度不能为负
            upper_bound = mean_val + threshold * std_val
            
            if method == 'clip':
                self.processed_data[col] = self.processed_data[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'remove':
                # 标记异常值但不删除（保持时间序列完整性）
                outliers_mask = (self.processed_data[col] < lower_bound) | (self.processed_data[col] > upper_bound)
                if outliers_mask.any():
                    logger.warning(f"列 {col} 发现 {outliers_mask.sum()} 个异常值，已用均值替换")
                    self.processed_data.loc[outliers_mask, col] = mean_val
            else:
                raise ValueError(f"不支持的异常值处理方法: {method}")
        
        logger.info("异常值处理完成")
        return self.processed_data
    
    def normalize_data(self, method: str = 'standard') -> pd.DataFrame:
        """
        数据标准化/归一化
        
        Args:
            method: 归一化方法，支持 'standard', 'minmax'
            
        Returns:
            标准化后的数据
        """
        if self.processed_data is None:
            raise ValueError("请先处理数据")
        
        logger.info(f"数据标准化，方法: {method}")
        
        # 创建标准化器
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        # 执行标准化
        traffic_data = self.processed_data[self.node_ids].values.T  # 形状: [num_nodes, time_steps]
        num_nodes, time_steps = traffic_data.shape
        traffic_data_flat = traffic_data.reshape(-1, 1)
        traffic_data_flat_scaled = self.scaler.fit_transform(traffic_data_flat)
        traffic_data_scaled = traffic_data_flat_scaled.reshape(num_nodes, time_steps)
        
        # 更新数据
        scaled_df = pd.DataFrame(traffic_data_scaled.T, columns=self.node_ids)
        self.processed_data[self.node_ids] = scaled_df
        
        logger.info("数据标准化完成")
        return self.processed_data
    
    def add_time_features(self) -> pd.DataFrame:
        """
        添加时间特征
        
        Returns:
            添加特征后的数据
        """
        if self.processed_data is None:
            raise ValueError("请先处理数据")
        
        logger.info("添加时间特征...")
        
        # 添加小时、工作日/周末特征
        self.processed_data['hour'] = self.processed_data['timestamp'].dt.hour
        self.processed_data['day_of_week'] = self.processed_data['timestamp'].dt.dayofweek
        self.processed_data['is_weekend'] = (self.processed_data['day_of_week'] >= 5).astype(int)
        
        # 添加时间周期的正弦和余弦特征（捕捉周期性）
        self.processed_data['hour_sin'] = np.sin(2 * np.pi * self.processed_data['hour'] / 24)
        self.processed_data['hour_cos'] = np.cos(2 * np.pi * self.processed_data['hour'] / 24)
        self.processed_data['weekday_sin'] = np.sin(2 * np.pi * self.processed_data['day_of_week'] / 7)
        self.processed_data['weekday_cos'] = np.cos(2 * np.pi * self.processed_data['day_of_week'] / 7)
        
        logger.info("时间特征添加完成")
        return self.processed_data
    
    def create_sequences(self, sequence_length: int = 12, prediction_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建序列数据
        
        Args:
            sequence_length: 输入序列长度
            prediction_steps: 预测步数
            
        Returns:
            X, y: 输入序列和目标序列
        """
        if self.processed_data is None:
            raise ValueError("请先处理数据")
        
        logger.info(f"创建序列数据，序列长度: {sequence_length}, 预测步数: {prediction_steps}")
        
        # 提取交通数据
        traffic_data = self.processed_data[self.node_ids].values.T  # [num_nodes, time_steps]
        
        # 创建序列
        X, y = [], []
        total_sequence_length = sequence_length + prediction_steps
        
        for i in range(traffic_data.shape[1] - total_sequence_length + 1):
            x_seq = traffic_data[:, i:i+sequence_length]
            y_seq = traffic_data[:, i+sequence_length:i+total_sequence_length]
            X.append(x_seq)
            y.append(y_seq)
        
        # 转换为张量并调整形状
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
        
        # 调整为 [num_samples, sequence_length, num_nodes, 1] 格式
        X_tensor = X_tensor.permute(0, 2, 1).unsqueeze(-1)
        y_tensor = y_tensor.permute(0, 2, 1).unsqueeze(-1)
        
        logger.info(f"序列创建完成，X形状: {X_tensor.shape}, y形状: {y_tensor.shape}")
        return X_tensor, y_tensor
    
    def split_data(self, X: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.7, val_ratio: float = 0.1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        分割训练/验证/测试集
        
        Args:
            X: 输入特征
            y: 目标值
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        logger.info(f"分割数据集，训练集比例: {train_ratio}, 验证集比例: {val_ratio}")
        
        # 计算分割大小
        total_size = len(X)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # 分割数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=self.random_seed
        )
        
        val_size_adjusted = int(len(X_temp) * (val_ratio / (1 - train_ratio)))
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_size_adjusted, random_state=self.random_seed
        )
        
        logger.info(f"数据集分割完成: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_processed_data(self, output_dir: str) -> Dict[str, str]:
        """
        保存处理后的数据
        
        Args:
            output_dir: 输出目录
            
        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建版本子目录
        version_dir = os.path.join(output_dir, self.version)
        os.makedirs(version_dir, exist_ok=True)
        
        file_paths = {}
        
        # 保存原始数据统计信息
        if self.raw_data is not None:
            stats_file = os.path.join(version_dir, f"{self.dataset_name}_stats.json")
            numeric_data = self.raw_data[self.node_ids]
            stats = {
                'mean': numeric_data.mean().to_dict(),
                'std': numeric_data.std().to_dict(),
                'min': numeric_data.min().to_dict(),
                'max': numeric_data.max().to_dict(),
                'shape': list(self.raw_data.shape)
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            file_paths['stats'] = stats_file
        
        # 保存标准化器
        if self.scaler is not None:
            scaler_file = os.path.join(version_dir, f"{self.dataset_name}_scaler.pkl")
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            file_paths['scaler'] = scaler_file
        
        # 保存处理后的数据
        if self.processed_data is not None:
            processed_file = os.path.join(version_dir, f"{self.dataset_name}_processed.csv")
            self.processed_data.to_csv(processed_file, index=False)
            file_paths['processed_data'] = processed_file
        
        # 保存元数据
        metadata = {
            'dataset_name': self.dataset_name,
            'version': self.version,
            'random_seed': self.random_seed,
            'num_nodes': self.num_nodes,
            'processing_time': datetime.now().isoformat(),
            'files': file_paths
        }
        metadata_file = os.path.join(version_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        file_paths['metadata'] = metadata_file
        
        # 更新latest链接
        latest_dir = os.path.join(output_dir, "latest")
        if os.path.exists(latest_dir):
            import shutil
            shutil.rmtree(latest_dir)
        os.symlink(version_dir, latest_dir) if hasattr(os, 'symlink') else None
        
        logger.info(f"数据保存完成，版本: {self.version}, 目录: {version_dir}")
        return file_paths
    
    def save_dataset_split(self, output_dir: str, datasets: Tuple, sequence_length: int, prediction_steps: int) -> Dict[str, str]:
        """
        保存数据集分割
        
        Args:
            output_dir: 输出目录
            datasets: (train, val, test) 数据集元组
            sequence_length: 序列长度
            prediction_steps: 预测步数
            
        Returns:
            保存的文件路径
        """
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
        
        # 确保版本目录存在
        version_dir = os.path.join(output_dir, self.version)
        os.makedirs(version_dir, exist_ok=True)
        
        # 创建序列数据目录
        seq_dir = os.path.join(version_dir, f"seq_len_{sequence_length}_pred_{prediction_steps}")
        os.makedirs(seq_dir, exist_ok=True)
        
        file_paths = {}
        
        # 保存训练集
        train_data = {
            'X': X_train.numpy(),
            'y': y_train.numpy()
        }
        train_file = os.path.join(seq_dir, "train_data.pkl")
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f)
        file_paths['train'] = train_file
        
        # 保存验证集
        val_data = {
            'X': X_val.numpy(),
            'y': y_val.numpy()
        }
        val_file = os.path.join(seq_dir, "val_data.pkl")
        with open(val_file, 'wb') as f:
            pickle.dump(val_data, f)
        file_paths['val'] = val_file
        
        # 保存测试集
        test_data = {
            'X': X_test.numpy(),
            'y': y_test.numpy()
        }
        test_file = os.path.join(seq_dir, "test_data.pkl")
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        file_paths['test'] = test_file
        
        # 保存分割配置
        split_config = {
            'sequence_length': sequence_length,
            'prediction_steps': prediction_steps,
            'random_seed': self.random_seed,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'split_time': datetime.now().isoformat()
        }
        config_file = os.path.join(seq_dir, "split_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(split_config, f, indent=2, ensure_ascii=False)
        file_paths['config'] = config_file
        
        logger.info(f"数据集分割保存完成: {seq_dir}")
        return file_paths
    
    def run_full_preprocessing(self, output_dir: str, sequence_length: int = 12, 
                              prediction_steps: int = 3, train_ratio: float = 0.7,
                              val_ratio: float = 0.1, missing_method: str = 'interpolate',
                              outlier_method: str = 'clip', normalize_method: str = 'standard',
                              add_time_features_flag: bool = True) -> Dict[str, any]:
        """
        运行完整的预处理流程
        
        Args:
            output_dir: 输出目录
            sequence_length: 序列长度
            prediction_steps: 预测步数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            missing_method: 缺失值处理方法
            outlier_method: 异常值处理方法
            normalize_method: 归一化方法
            add_time_features_flag: 是否添加时间特征
            
        Returns:
            预处理结果报告
        """
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 数据质量检查
            quality_report = self.check_data_quality()
            
            # 3. 处理缺失值
            self.handle_missing_values(method=missing_method)
            
            # 4. 处理异常值
            self.handle_outliers(method=outlier_method)
            
            # 5. 添加时间特征
            if add_time_features_flag:
                self.add_time_features()
            
            # 6. 数据标准化
            self.normalize_data(method=normalize_method)
            
            # 7. 创建序列数据
            X, y = self.create_sequences(sequence_length, prediction_steps)
            
            # 8. 分割数据集
            datasets = self.split_data(X, y, train_ratio, val_ratio)
            
            # 9. 保存处理后的数据
            processed_files = self.save_processed_data(output_dir)
            
            # 10. 保存数据集分割
            split_files = self.save_dataset_split(output_dir, datasets, sequence_length, prediction_steps)
            
            # 生成完整报告
            report = {
                'version': self.version,
                'dataset_name': self.dataset_name,
                'random_seed': self.random_seed,
                'processing_parameters': {
                    'sequence_length': sequence_length,
                    'prediction_steps': prediction_steps,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'missing_method': missing_method,
                    'outlier_method': outlier_method,
                    'normalize_method': normalize_method,
                    'add_time_features': add_time_features_flag
                },
                'quality_report': quality_report,
                'file_paths': {
                    'processed': processed_files,
                    'split': split_files
                }
            }
            
            # 保存报告
            report_file = os.path.join(output_dir, self.version, "preprocessing_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"完整预处理流程完成，报告保存至: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"预处理过程中出错: {e}")
            raise

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='交通数据预处理脚本')
    parser.add_argument('--dataset', type=str, default='METR-LA', choices=['METR-LA', 'PEMS-BAY'],
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='d:\gcn-lstm\data\real-data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='d:\gcn-lstm\data\processed',
                        help='输出目录路径')
    parser.add_argument('--sequence_length', type=int, default=12,
                        help='输入序列长度')
    parser.add_argument('--prediction_steps', type=int, default=3,
                        help='预测步数')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--missing_method', type=str, default='interpolate',
                        choices=['interpolate', 'ffill', 'bfill', 'mean'],
                        help='缺失值处理方法')
    parser.add_argument('--outlier_method', type=str, default='clip',
                        choices=['clip', 'remove'],
                        help='异常值处理方法')
    parser.add_argument('--normalize_method', type=str, default='standard',
                        choices=['standard', 'minmax'],
                        help='归一化方法')
    parser.add_argument('--no_time_features', action='store_true',
                        help='不添加时间特征')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    logger.info(f"开始数据预处理，数据集: {args.dataset}")
    logger.info(f"参数配置: {vars(args)}")
    
    # 创建预处理器
    preprocessor = TrafficDataPreprocessor(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        random_seed=args.random_seed
    )
    
    # 运行预处理
    report = preprocessor.run_full_preprocessing(
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        prediction_steps=args.prediction_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        missing_method=args.missing_method,
        outlier_method=args.outlier_method,
        normalize_method=args.normalize_method,
        add_time_features_flag=not args.no_time_features
    )
    
    logger.info("数据预处理完成！")
    logger.info(f"版本: {report['version']}")
    logger.info(f"训练集大小: {report['processing_parameters']['train_ratio']}")
    logger.info(f"验证集大小: {report['processing_parameters']['val_ratio']}")
    logger.info(f"测试集大小: {1 - report['processing_parameters']['train_ratio'] - report['processing_parameters']['val_ratio']}")
    
    # 打印主要文件路径
    print("\n预处理完成，生成的文件:")
    print(f"- 报告文件: {report['file_paths']['processed']['metadata']}")
    print(f"- 训练数据: {report['file_paths']['split']['train']}")
    print(f"- 验证数据: {report['file_paths']['split']['val']}")
    print(f"- 测试数据: {report['file_paths']['split']['test']}")

if __name__ == "__main__":
    main()