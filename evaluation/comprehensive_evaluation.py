"""
综合评估脚本

此脚本提供统一的模型评估框架，用于：
- 加载和评估多个模型（包括基线模型）
- 多次运行以计算统计显著性
- 保存详细的评估结果
- 记录评估环境信息
- 生成可视化报告

使用方法：
    python comprehensive_evaluation.py --config ../configs/experiment_config.yaml --output_dir ../results
"""

import os
import sys
import argparse
import json
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import platform
import psutil
from typing import Dict, List, Tuple, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入配置加载器
from code.config.config_loader import ConfigLoader

# 导入数据加载功能
from code.data_integration.real_traffic_data_loader import RealTrafficDataLoader

# 导入模型（这里假设模型已经实现）
from code.models.gcn_lstm_hybrid import GCNLSTMHybrid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('comprehensive_evaluator')

# 评估指标函数
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测值 [samples, steps, nodes, 1]
        targets: 真实值 [samples, steps, nodes, 1]
        
    Returns:
        评估指标字典
    """
    # 展平数据
    y_pred = predictions.reshape(-1)
    y_true = targets.reshape(-1)
    
    # 避免除零错误
    mask = y_true > 0
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    
    # 计算MAE
    mae = np.mean(np.abs(y_pred - y_true))
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # 计算MAPE
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    
    # 计算R²
    if len(y_true) > 1:
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    else:
        r2 = 0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2)
    }

def calculate_statistical_significance(results: Dict[str, List[Dict[str, float]]], 
                                     baseline_model: str) -> Dict[str, Dict[str, float]]:
    """
    计算统计显著性（t检验）
    
    Args:
        results: 各模型多次运行的结果
        baseline_model: 基线模型名称
        
    Returns:
        统计显著性结果
    """
    from scipy import stats
    
    significance_results = {}
    
    if baseline_model not in results:
        logger.warning(f"基线模型 {baseline_model} 不在结果中")
        return significance_results
    
    baseline_scores = {}
    # 收集基线模型的指标
    for metric in results[baseline_model][0].keys():
        baseline_scores[metric] = [run[metric] for run in results[baseline_model]]
    
    # 对比其他模型
    for model_name, model_runs in results.items():
        if model_name == baseline_model:
            continue
        
        model_significance = {}
        for metric in baseline_scores.keys():
            # 收集当前模型的指标
            model_scores = [run[metric] for run in model_runs]
            
            # 执行配对t检验（假设多次运行是成对的）
            try:
                _, p_value = stats.ttest_rel(baseline_scores[metric], model_scores)
                model_significance[metric] = float(p_value)
            except Exception as e:
                logger.warning(f"计算 {metric} 的统计显著性失败: {e}")
                model_significance[metric] = None
        
        significance_results[model_name] = model_significance
    
    return significance_results

class ModelEvaluator:
    """
    模型评估器类
    """
    
    def __init__(self, config_path: str, output_dir: str):
        """
        初始化评估器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.loader = ConfigLoader()
        self.config = self.loader.load_config(config_path)
        
        # 创建输出目录
        self.eval_dir = os.path.join(
            output_dir,
            f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config.get('environment', {}).get('device') == 'cuda' else 'cpu'
        )
        
        # 记录环境信息
        self.environment_info = self._collect_environment_info()
        
        logger.info(f"评估器初始化完成，设备: {self.device}")
        logger.info(f"输出目录: {self.eval_dir}")
    
    def _collect_environment_info(self) -> Dict[str, str]:
        """
        收集环境信息
        
        Returns:
            环境信息字典
        """
        info = {
            'python_version': platform.python_version(),
            'os': platform.system() + ' ' + platform.release(),
            'cpu': platform.processor(),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'torch_version': torch.__version__ if torch.cuda.is_available() else 'N/A',
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'evaluation_time': datetime.now().isoformat()
        }
        
        # 添加CUDA设备信息
        if torch.cuda.is_available():
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        return info
    
    def load_test_data(self) -> Tuple[torch.utils.data.DataLoader, Any]:
        """
        加载测试数据
        
        Returns:
            测试数据加载器和邻接矩阵
        """
        logger.info("加载测试数据...")
        
        # 获取数据集配置
        dataset_config = self.config.get('dataset', {})
        dataset_name = dataset_config.get('name', 'METR-LA')
        data_dir = dataset_config.get('data_dir', 'd:/gcn-lstm/data/real-data')
        batch_size = dataset_config.get('batch_size', 32)
        sequence_length = dataset_config.get('sequence_length', 12)
        prediction_steps = dataset_config.get('prediction_steps', 3)
        
        # 初始化数据加载器
        data_loader = RealTrafficDataLoader(data_dir=data_dir, dataset_name=dataset_name)
        
        # 获取测试数据
        _, _, test_loader = data_loader.get_data_loaders(
            batch_size=batch_size,
            sequence_length=sequence_length,
            prediction_steps=prediction_steps,
            shuffle=False
        )
        
        # 获取邻接矩阵
        adj_matrix = data_loader.build_adj_matrix()
        
        logger.info(f"测试数据加载完成，批次数量: {len(test_loader)}")
        return test_loader, adj_matrix
    
    def load_model(self, model_path: str, model_type: str = 'gcn_lstm_hybrid', adj_matrix: torch.Tensor = None) -> nn.Module:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型
            adj_matrix: 邻接矩阵（如果需要）
            
        Returns:
            加载的模型
        """
        logger.info(f"加载模型: {model_path}, 类型: {model_type}")
        
        try:
            # 根据模型类型加载不同的模型
            if model_type == 'gcn_lstm_hybrid':
                # 获取模型配置
                model_config = self.loader.get_model_config()
                
                # 创建模型实例
                model = GCNLSTMHybrid(config=model_config)
                
                # 加载模型权重
                state_dict = torch.load(model_path, map_location=self.device)
                
                # 处理可能的DataParallel包装
                if 'module.' in list(state_dict.keys())[0]:
                    # 创建新的state_dict，去掉'module.'前缀
                    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)
                
                # 设置为评估模式
                model.eval()
                model.to(self.device)
                
                logger.info("GCN-LSTM混合模型加载完成")
                return model
                
            elif model_type == 'lstm':
                # 这里应该加载LSTM模型
                # 暂时返回模拟模型
                logger.warning("LSTM模型加载功能待实现")
                return self._create_dummy_model()
                
            elif model_type == 'gcn':
                # 这里应该加载GCN模型
                # 暂时返回模拟模型
                logger.warning("GCN模型加载功能待实现")
                return self._create_dummy_model()
                
            elif model_type == 'stgcn':
                # 这里应该加载STGCN模型
                # 暂时返回模拟模型
                logger.warning("STGCN模型加载功能待实现")
                return self._create_dummy_model()
                
            elif model_type == 't-gcn':
                # 这里应该加载T-GCN模型
                # 暂时返回模拟模型
                logger.warning("T-GCN模型加载功能待实现")
                return self._create_dummy_model()
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _create_dummy_model(self) -> nn.Module:
        """
        创建虚拟模型用于测试
        
        Returns:
            虚拟模型
        """
        # 创建一个简单的模型用于测试评估流程
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, adj_matrix=None):
                # 返回输入的随机扰动作为预测
                noise = torch.randn_like(x) * 0.1
                return x + noise
        
        model = DummyModel()
        model.eval()
        model.to(self.device)
        return model
    
    def evaluate_model(self, model: nn.Module, test_loader: torch.utils.data.DataLoader, 
                      adj_matrix: torch.Tensor = None) -> Dict[str, Any]:
        """
        评估单个模型
        
        Args:
            model: 要评估的模型
            test_loader: 测试数据加载器
            adj_matrix: 邻接矩阵（如果需要）
            
        Returns:
            评估结果字典
        """
        logger.info("开始评估模型...")
        
        all_predictions = []
        all_targets = []
        inference_times = []
        
        # 将邻接矩阵移至设备
        if adj_matrix is not None:
            adj_matrix = adj_matrix.to(self.device)
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(test_loader):
                # 将数据移至设备
                X = X.to(self.device)
                y = y.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                
                # 前向传播
                if 'gcn' in str(type(model)).lower():
                    # GCN相关模型需要邻接矩阵
                    outputs = model(X, adj_matrix)
                else:
                    outputs = model(X)
                
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                inference_times.append(inference_time)
                
                # 收集预测结果和真实值
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"处理批次 {batch_idx + 1}/{len(test_loader)}")
        
        # 合并所有批次的结果
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 计算指标
        metrics = calculate_metrics(predictions, targets)
        
        # 计算推理时间统计
        metrics['inference_time_ms_avg'] = float(np.mean(inference_times))
        metrics['inference_time_ms_std'] = float(np.std(inference_times))
        metrics['inference_time_ms_min'] = float(np.min(inference_times))
        metrics['inference_time_ms_max'] = float(np.max(inference_times))
        
        logger.info(f"模型评估完成，MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'inference_times': inference_times
        }
    
    def run_evaluation(self, model_paths: Dict[str, str], baseline_model: str = 'lstm') -> Dict[str, Any]:
        """
        运行完整评估流程
        
        Args:
            model_paths: 模型名称到模型路径的映射
            baseline_model: 基线模型名称
            
        Returns:
            评估报告
        """
        logger.info(f"开始综合评估，模型数量: {len(model_paths)}")
        
        # 加载测试数据
        test_loader, adj_matrix = self.load_test_data()
        
        # 获取评估配置
        eval_config = self.loader.get_evaluation_config()
        run_times = eval_config.get('run_times', 5)
        
        # 保存配置
        self.loader.save_config(os.path.join(self.eval_dir, 'evaluation_config.yaml'))
        
        # 保存环境信息
        with open(os.path.join(self.eval_dir, 'environment_info.json'), 'w', encoding='utf-8') as f:
            json.dump(self.environment_info, f, indent=2, ensure_ascii=False)
        
        # 评估结果字典
        all_results = {}
        all_predictions = {}
        all_targets = {}
        
        # 对每个模型进行评估
        for model_name, model_path in model_paths.items():
            logger.info(f"评估模型: {model_name}")
            
            # 确定模型类型
            model_type = self._get_model_type(model_name)
            
            # 多次运行
            run_results = []
            
            for run_idx in range(run_times):
                logger.info(f"运行 {run_idx + 1}/{run_times}")
                
                # 加载模型
                model = self.load_model(model_path, model_type, adj_matrix)
                
                # 评估模型
                result = self.evaluate_model(model, test_loader, adj_matrix)
                
                # 保存运行结果
                run_results.append(result['metrics'])
                
                # 保存第一次运行的预测结果用于可视化
                if run_idx == 0:
                    all_predictions[model_name] = result['predictions']
                    all_targets[model_name] = result['targets']
            
            # 计算多次运行的统计信息
            aggregated_results = self._aggregate_results(run_results)
            all_results[model_name] = aggregated_results
            
            logger.info(f"模型 {model_name} 评估完成，平均MAE: {aggregated_results['mean']['mae']:.4f}")
        
        # 计算统计显著性
        significance_results = calculate_statistical_significance(
            {model: results['runs'] for model, results in all_results.items()}, 
            baseline_model
        )
        
        # 生成综合报告
        report = {
            'evaluation_time': datetime.now().isoformat(),
            'environment': self.environment_info,
            'config': self.config,
            'models': all_results,
            'statistical_significance': significance_results,
            'run_times': run_times,
            'baseline_model': baseline_model
        }
        
        # 保存报告
        report_path = os.path.join(self.eval_dir, 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存详细的运行结果
        runs_path = os.path.join(self.eval_dir, 'run_results.json')
        runs_data = {model: results['runs'] for model, results in all_results.items()}
        with open(runs_path, 'w', encoding='utf-8') as f:
            json.dump(runs_data, f, indent=2, ensure_ascii=False)
        
        # 生成可视化
        if eval_config.get('visualize_results', True):
            self._generate_visualizations(all_results, all_predictions, all_targets)
        
        logger.info(f"综合评估完成，报告保存至: {report_path}")
        return report
    
    def _get_model_type(self, model_name: str) -> str:
        """
        根据模型名称确定模型类型
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型类型
        """
        model_name_lower = model_name.lower()
        
        if 'lstm' in model_name_lower and 'gcn' in model_name_lower:
            return 'gcn_lstm_hybrid'
        elif 'lstm' in model_name_lower:
            return 'lstm'
        elif 'gcn' in model_name_lower:
            return 'gcn'
        elif 'stgcn' in model_name_lower:
            return 'stgcn'
        elif 't-gcn' in model_name_lower or 'tgcn' in model_name_lower:
            return 't-gcn'
        else:
            logger.warning(f"未知模型类型: {model_name}，使用默认类型")
            return 'gcn_lstm_hybrid'
    
    def _aggregate_results(self, run_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        聚合多次运行的结果
        
        Args:
            run_results: 多次运行的结果列表
            
        Returns:
            聚合后的结果
        """
        aggregated = {
            'mean': {},
            'std': {},
            'min': {},
            'max': {},
            'runs': run_results
        }
        
        # 获取所有指标名称
        metrics = run_results[0].keys() if run_results else []
        
        for metric in metrics:
            values = [run[metric] for run in run_results]
            aggregated['mean'][metric] = float(np.mean(values))
            aggregated['std'][metric] = float(np.std(values))
            aggregated['min'][metric] = float(np.min(values))
            aggregated['max'][metric] = float(np.max(values))
        
        return aggregated
    
    def _generate_visualizations(self, results: Dict[str, Any], 
                               predictions: Dict[str, np.ndarray],
                               targets: Dict[str, np.ndarray]):
        """
        生成可视化结果
        
        Args:
            results: 评估结果
            predictions: 预测结果
            targets: 真实值
        """
        viz_dir = os.path.join(self.eval_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 设置可视化样式
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. 绘制指标对比图
        self._plot_metrics_comparison(results, viz_dir)
        
        # 2. 绘制箱线图
        self._plot_boxplots(results, viz_dir)
        
        # 3. 绘制预测vs真实值示例
        if predictions and targets:
            self._plot_prediction_examples(predictions, targets, viz_dir)
        
        logger.info(f"可视化完成，保存至: {viz_dir}")
    
    def _plot_metrics_comparison(self, results: Dict[str, Any], viz_dir: str):
        """
        绘制指标对比图
        """
        metrics_to_plot = ['mae', 'rmse', 'mape', 'r2']
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            model_names = []
            mean_values = []
            std_values = []
            
            for model_name, model_results in results.items():
                model_names.append(model_name)
                mean_values.append(model_results['mean'][metric])
                std_values.append(model_results['std'][metric])
            
            # 绘制柱状图
            bars = plt.bar(model_names, mean_values, yerr=std_values, capsize=5)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, 
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.title(f'{metric.upper()} Comparison')
            plt.xlabel('Model')
            plt.ylabel(metric.upper())
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(os.path.join(viz_dir, f"{metric}_comparison.png"))
            plt.close()
    
    def _plot_boxplots(self, results: Dict[str, Any], viz_dir: str):
        """
        绘制箱线图
        """
        metrics_to_plot = ['mae', 'rmse']
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            data_to_plot = []
            model_names = []
            
            for model_name, model_results in results.items():
                model_names.append(model_name)
                # 提取多次运行的指标值
                values = [run[metric] for run in model_results['runs']]
                data_to_plot.append(values)
            
            # 绘制箱线图
            box = plt.boxplot(data_to_plot, labels=model_names, patch_artist=True)
            
            # 设置箱体颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            plt.title(f'{metric.upper()} Distribution Across Runs')
            plt.xlabel('Model')
            plt.ylabel(metric.upper())
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(os.path.join(viz_dir, f"{metric}_boxplot.png"))
            plt.close()
    
    def _plot_prediction_examples(self, predictions: Dict[str, np.ndarray],
                                 targets: Dict[str, np.ndarray], viz_dir: str):
        """
        绘制预测vs真实值示例
        """
        # 选择一个样本进行可视化
        sample_idx = 0
        node_idx = 0
        steps = min(24, predictions[list(predictions.keys())[0]].shape[1])
        
        plt.figure(figsize=(12, 6))
        
        # 绘制真实值
        time_steps = np.arange(steps)
        for model_name, pred in predictions.items():
            # 只绘制前几个时间步
            plt.plot(time_steps, pred[sample_idx, :steps, node_idx, 0], 
                    label=f'{model_name} Prediction')
        
        # 绘制所有模型的真实值（应该相同）
        first_model = list(targets.keys())[0]
        plt.plot(time_steps, targets[first_model][sample_idx, :steps, node_idx, 0], 
                'k-', label='True Value', linewidth=2)
        
        plt.title(f'Prediction vs True Value (Sample {sample_idx}, Node {node_idx})')
        plt.xlabel('Time Step')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(viz_dir, "prediction_example.png"))
        plt.close()

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='综合评估脚本')
    parser.add_argument('--config', type=str, default='../configs/experiment_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='输出目录路径')
    parser.add_argument('--baseline', type=str, default='lstm',
                        help='基线模型名称')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['gcn_lstm_hybrid', 'lstm', 'gcn', 'stgcn', 't-gcn'],
                        help='要评估的模型列表')
    parser.add_argument('--model_paths', type=str, default=None,
                        help='模型路径配置文件（JSON格式）')
    return parser.parse_args()

def get_default_model_paths():
    """
    获取默认模型路径
    
    Returns:
        模型路径字典
    """
    # 这里应该返回真实的模型路径
    # 暂时使用虚拟路径
    return {
        'gcn_lstm_hybrid': 'd:/gcn-lstm/models/saved/gcn_lstm_hybrid_model.pth',
        'lstm': 'd:/gcn-lstm/models/saved/lstm_model.pth',
        'gcn': 'd:/gcn-lstm/models/saved/gcn_model.pth',
        'stgcn': 'd:/gcn-lstm/models/saved/stgcn_model.pth',
        't-gcn': 'd:/gcn-lstm/models/saved/tgcn_model.pth'
    }

def main():
    """
    主函数
    """
    args = parse_args()
    
    logger.info("开始综合评估")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 加载模型路径
    if args.model_paths and os.path.exists(args.model_paths):
        with open(args.model_paths, 'r', encoding='utf-8') as f:
            model_paths = json.load(f)
    else:
        model_paths = get_default_model_paths()
        logger.warning("未找到模型路径配置文件，使用默认路径")
    
    # 过滤指定的模型
    if args.models:
        filtered_model_paths = {}
        for model_name in args.models:
            if model_name in model_paths:
                filtered_model_paths[model_name] = model_paths[model_name]
            else:
                logger.warning(f"模型 {model_name} 路径未找到")
        model_paths = filtered_model_paths
    
    # 创建评估器
    evaluator = ModelEvaluator(args.config, args.output_dir)
    
    # 运行评估
    report = evaluator.run_evaluation(model_paths, args.baseline)
    
    # 打印摘要
    print("\n评估完成！")
    print(f"评估目录: {evaluator.eval_dir}")
    print("\n模型性能摘要:")
    for model_name, results in report['models'].items():
        print(f"{model_name}:")
        print(f"  MAE: {results['mean']['mae']:.4f} ± {results['std']['mae']:.4f}")
        print(f"  RMSE: {results['mean']['rmse']:.4f} ± {results['std']['rmse']:.4f}")
        print(f"  平均推理时间: {results['mean']['inference_time_ms_avg']:.2f} ms")
    
    if report['statistical_significance']:
        print("\n统计显著性结果 (p值):")
        for model_name, sig_results in report['statistical_significance'].items():
            print(f"{model_name} vs {report['baseline_model']}:")
            for metric, p_value in sig_results.items():
                significance = "显著" if p_value < 0.05 else "不显著"
                print(f"  {metric}: {p_value:.4f} ({significance})")

if __name__ == "__main__":
    main()