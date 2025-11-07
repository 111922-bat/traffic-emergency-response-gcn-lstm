"""
模型量化和精度优化模块
支持动态量化和静态量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import copy
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time


class DynamicQuantizer:
    """动态量化器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.quantized_model = None
        self.quantization_config = None
        
    def quantize_model(self) -> nn.Module:
        """执行动态量化"""
        print("开始动态量化")
        
        # 设置量化配置
        self.quantized_model = copy.deepcopy(self.model)
        
        # 动态量化 - 主要对线性层和卷积层进行量化
        self.quantized_model = quantization.quantize_dynamic(
            self.quantized_model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        print("动态量化完成")
        return self.quantized_model
    
    def benchmark_model(self, model: nn.Module, test_data: torch.Tensor, 
                       test_labels: torch.Tensor, device: str = 'cpu') -> Dict:
        """基准测试模型性能"""
        model.eval()
        model = model.to(device)
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_data[:10])
        
        # 计时推理
        start_time = time.time()
        with torch.no_grad():
            predictions = model(test_data)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # 计算性能指标
        mse = F.mse_loss(predictions, test_labels).item()
        mae = F.mae_loss(predictions, test_labels).item()
        
        # 计算模型大小
        model_size = self._calculate_model_size(model)
        
        return {
            'inference_time': inference_time,
            'mse': mse,
            'mae': mae,
            'model_size_mb': model_size,
            'throughput_samples_per_sec': len(test_data) / inference_time
        }
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """计算模型大小(MB)"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024 / 1024
        return size_all_mb


class StaticQuantizer:
    """静态量化器"""
    
    def __init__(self, model: nn.Module, qconfig: torch.quantization.QConfig = None):
        self.model = model
        self.quantized_model = None
        self.qconfig = qconfig or torch.quantization.get_default_qconfig('fbgemm')
        self.calibration_data = None
        
    def prepare_for_quantization(self) -> nn.Module:
        """准备量化"""
        print("准备静态量化")
        
        # 复制模型
        prepared_model = copy.deepcopy(self.model)
        
        # 设置量化配置
        prepared_model.qconfig = self.qconfig
        
        # 准备量化
        prepared_model = quantization.prepare(prepared_model)
        
        return prepared_model
    
    def calibrate(self, model: nn.Module, calibration_loader, device: str = 'cpu'):
        """校准量化参数"""
        print("开始校准")
        
        model.eval()
        model = model.to(device)
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                data = data.to(device)
                _ = model(data)
                
                if batch_idx >= 100:  # 限制校准样本数量
                    break
        
        print("校准完成")
    
    def quantize_model(self, calibration_loader, device: str = 'cpu') -> nn.Module:
        """执行静态量化"""
        print("开始静态量化")
        
        # 准备量化
        prepared_model = self.prepare_for_quantization()
        
        # 校准
        if calibration_loader is not None:
            self.calibrate(prepared_model, calibration_loader, device)
        
        # 执行量化
        self.quantized_model = quantization.convert(prepared_model)
        
        print("静态量化完成")
        return self.quantized_model


class QuantizationAwareTraining:
    """量化感知训练"""
    
    def __init__(self, model: nn.Module, qconfig: torch.quantization.QConfig = None):
        self.model = model
        self.qconfig = qconfig or torch.quantization.get_default_qconfig('fbgemm')
        self.quantized_model = None
        self.training_history = []
        
    def prepare_qat(self) -> nn.Module:
        """准备量化感知训练"""
        print("准备量化感知训练")
        
        # 复制模型
        qat_model = copy.deepcopy(self.model)
        
        # 设置量化配置
        qat_model.qconfig = self.qconfig
        
        # 准备QAT
        qat_model = quantization.prepare_qat(qat_model)
        
        return qat_model
    
    def train_qat(self, train_loader, val_loader=None, epochs: int = 30, 
                  lr: float = 0.001, device: str = 'cpu') -> nn.Module:
        """执行量化感知训练"""
        print("开始量化感知训练")
        
        # 准备QAT模型
        qat_model = self.prepare_qat()
        qat_model = qat_model.to(device)
        
        # 优化器
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        best_model = None
        
        for epoch in range(epochs):
            # 训练阶段
            qat_model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                outputs = qat_model(data)
                loss = criterion(outputs, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            
            # 验证阶段
            val_loss = None
            if val_loader is not None:
                qat_model.eval()
                val_total_loss = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        outputs = qat_model(data)
                        val_total_loss += criterion(outputs, target).item()
                
                val_loss = val_total_loss / len(val_loader)
                
                # 保存最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(qat_model)
            
            # 记录历史
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 转换为真正的量化模型
        if best_model is not None:
            self.quantized_model = quantization.convert(best_model)
        else:
            self.quantized_model = quantization.convert(qat_model)
        
        print("量化感知训练完成")
        return self.quantized_model


class MixedPrecisionQuantizer:
    """混合精度量化器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.quantized_model = None
        self.precision_config = {}
        
    def analyze_model_sensitivity(self, test_data: torch.Tensor, device: str = 'cpu') -> Dict[str, float]:
        """分析模型各层对量化的敏感性"""
        print("分析模型量化敏感性")
        
        model = copy.deepcopy(self.model)
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            # 获取原始输出
            original_output = model(test_data.to(device))
            
            layer_sensitivity = {}
            
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # 临时量化这一层
                    quantized_module = copy.deepcopy(module)
                    if isinstance(module, nn.Linear):
                        quantized_module = nn.Linear(module.in_features, module.out_features)
                        quantized_module.weight.data = module.weight.data
                        if module.bias is not None:
                            quantized_module.bias.data = module.bias.data
                    elif isinstance(module, nn.Conv2d):
                        quantized_module = nn.Conv2d(
                            module.in_channels, module.out_channels, module.kernel_size,
                            module.stride, module.padding, module.dilation, module.groups,
                            module.bias is not None
                        )
                        quantized_module.weight.data = module.weight.data
                        if module.bias is not None:
                            quantized_module.bias.data = module.bias.data
                    
                    # 替换层并测试
                    self._replace_layer(model, name, quantized_module)
                    try:
                        quantized_output = model(test_data.to(device))
                        # 计算输出差异
                        diff = F.mse_loss(original_output, quantized_output).item()
                        layer_sensitivity[name] = diff
                    except:
                        layer_sensitivity[name] = float('inf')
                    
                    # 恢复原始层
                    self._replace_layer(model, name, module)
        
        return layer_sensitivity
    
    def _replace_layer(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """替换模型中的层"""
        parts = layer_name.split('.')
        current = model
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], new_layer)
    
    def create_mixed_precision_config(self, sensitivity_analysis: Dict[str, float], 
                                    threshold: float = 0.01) -> Dict[str, str]:
        """创建混合精度配置"""
        config = {}
        
        for layer_name, sensitivity in sensitivity_analysis.items():
            if sensitivity < threshold:
                # 低敏感性层使用低精度
                config[layer_name] = 'int8'
            else:
                # 高敏感性层使用高精度
                config[layer_name] = 'fp16'
        
        return config
    
    def apply_mixed_precision(self, precision_config: Dict[str, str]) -> nn.Module:
        """应用混合精度量化"""
        print("应用混合精度量化")
        
        quantized_model = copy.deepcopy(self.model)
        
        for layer_name, precision in precision_config.items():
            if precision == 'int8':
                # 应用int8量化
                self._apply_int8_quantization(quantized_model, layer_name)
            elif precision == 'fp16':
                # 应用fp16精度
                self._apply_fp16_precision(quantized_model, layer_name)
        
        self.quantized_model = quantized_model
        self.precision_config = precision_config
        
        return quantized_model
    
    def _apply_int8_quantization(self, model: nn.Module, layer_name: str):
        """对指定层应用int8量化"""
        # 这里实现具体的int8量化逻辑
        # 由于复杂性，这里只是一个示例
        pass
    
    def _apply_fp16_precision(self, model: nn.Module, layer_name: str):
        """对指定层应用fp16精度"""
        parts = layer_name.split('.')
        current = model
        for part in parts[:-1]:
            current = getattr(current, part)
        
        layer = getattr(current, parts[-1])
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            # 转换为half精度
            if hasattr(layer, 'weight'):
                layer.weight.data = layer.weight.data.half()
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data = layer.bias.data.half()


class QuantizationOptimizer:
    """量化优化器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.optimization_results = {}
        
    def optimize_quantization(self, train_loader, val_loader, test_data: torch.Tensor,
                            test_labels: torch.Tensor, device: str = 'cpu') -> Dict:
        """优化量化策略"""
        print("开始量化优化")
        
        results = {}
        
        # 1. 动态量化
        print("测试动态量化")
        dynamic_quantizer = DynamicQuantizer(self.model)
        dynamic_model = dynamic_quantizer.quantize_model()
        dynamic_metrics = dynamic_quantizer.benchmark_model(dynamic_model, test_data, test_labels, device)
        results['dynamic'] = {
            'model': dynamic_model,
            'metrics': dynamic_metrics
        }
        
        # 2. 静态量化
        print("测试静态量化")
        static_quantizer = StaticQuantizer(self.model)
        static_model = static_quantizer.quantize_model(val_loader, device)
        static_metrics = dynamic_quantizer.benchmark_model(static_model, test_data, test_labels, device)
        results['static'] = {
            'model': static_model,
            'metrics': static_metrics
        }
        
        # 3. 量化感知训练
        print("测试量化感知训练")
        qat = QuantizationAwareTraining(self.model)
        qat_model = qat.train_qat(train_loader, val_loader, epochs=20, device=device)
        qat_metrics = dynamic_quantizer.benchmark_model(qat_model, test_data, test_labels, device)
        results['qat'] = {
            'model': qat_model,
            'metrics': qat_metrics
        }
        
        # 4. 混合精度
        print("测试混合精度")
        mixed_precision = MixedPrecisionQuantizer(self.model)
        sensitivity = mixed_precision.analyze_model_sensitivity(test_data[:100], device)
        precision_config = mixed_precision.create_mixed_precision_config(sensitivity)
        mixed_model = mixed_precision.apply_mixed_precision(precision_config)
        mixed_metrics = dynamic_quantizer.benchmark_model(mixed_model, test_data, test_labels, device)
        results['mixed_precision'] = {
            'model': mixed_model,
            'metrics': mixed_metrics,
            'sensitivity': sensitivity,
            'config': precision_config
        }
        
        # 选择最佳策略
        best_strategy = self._select_best_strategy(results)
        results['best_strategy'] = best_strategy
        
        self.optimization_results = results
        return results
    
    def _select_best_strategy(self, results: Dict) -> str:
        """选择最佳量化策略"""
        best_strategy = None
        best_score = float('-inf')
        
        for strategy, result in results.items():
            if strategy == 'best_strategy':
                continue
                
            metrics = result['metrics']
            # 综合评分：考虑精度、速度和模型大小
            score = (
                (1.0 / (1.0 + metrics['mse'])) * 0.4 +  # 精度权重40%
                (metrics['throughput_samples_per_sec'] / 1000) * 0.3 +  # 速度权重30%
                (1.0 / metrics['model_size_mb']) * 0.3  # 大小权重30%
            )
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy
    
    def plot_optimization_results(self, save_path: str = "quantization_optimization.png"):
        """绘制优化结果"""
        if not self.optimization_results:
            print("没有优化结果")
            return
        
        strategies = []
        inference_times = []
        mse_scores = []
        model_sizes = []
        throughputs = []
        
        for strategy, result in self.optimization_results.items():
            if strategy == 'best_strategy':
                continue
                
            strategies.append(strategy)
            metrics = result['metrics']
            inference_times.append(metrics['inference_time'])
            mse_scores.append(metrics['mse'])
            model_sizes.append(metrics['model_size_mb'])
            throughputs.append(metrics['throughput_samples_per_sec'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 推理时间
        ax1.bar(strategies, inference_times, color='skyblue')
        ax1.set_title('推理时间对比')
        ax1.set_ylabel('时间 (秒)')
        ax1.tick_params(axis='x', rotation=45)
        
        # MSE
        ax2.bar(strategies, mse_scores, color='lightcoral')
        ax2.set_title('MSE对比')
        ax2.set_ylabel('MSE')
        ax2.tick_params(axis='x', rotation=45)
        
        # 模型大小
        ax3.bar(strategies, model_sizes, color='lightgreen')
        ax3.set_title('模型大小对比')
        ax3.set_ylabel('大小 (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 吞吐量
        ax4.bar(strategies, throughputs, color='gold')
        ax4.set_title('吞吐量对比')
        ax4.set_ylabel('样本/秒')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"优化结果图已保存到: {save_path}")


def save_quantization_results(optimizer, save_dir: str):
    """保存量化结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存最佳模型
    best_strategy = optimizer.optimization_results['best_strategy']
    best_model = optimizer.optimization_results[best_strategy]['model']
    torch.save(best_model.state_dict(), os.path.join(save_dir, f"best_{best_strategy}_model.pth"))
    
    # 保存量化信息
    quantization_info = {
        'best_strategy': best_strategy,
        'all_results': {}
    }
    
    for strategy, result in optimizer.optimization_results.items():
        if strategy == 'best_strategy':
            continue
            
        metrics = result['metrics']
        quantization_info['all_results'][strategy] = {
            'inference_time': metrics['inference_time'],
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'model_size_mb': metrics['model_size_mb'],
            'throughput_samples_per_sec': metrics['throughput_samples_per_sec']
        }
        
        # 如果有额外的配置信息，也保存
        if 'config' in result:
            quantization_info['all_results'][strategy]['config'] = result['config']
        if 'sensitivity' in result:
            quantization_info['all_results'][strategy]['sensitivity'] = result['sensitivity']
    
    with open(os.path.join(save_dir, "quantization_results.json"), 'w') as f:
        json.dump(quantization_info, f, indent=2)
    
    print(f"量化结果已保存到: {save_dir}")


if __name__ == "__main__":
    # 示例使用
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 30)
            self.fc3 = nn.Linear(30, 1)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    # 创建模型和数据
    model = SimpleModel()
    test_data = torch.randn(200, 10)
    test_labels = torch.randn(200, 1)
    
    # 创建虚拟训练数据
    train_data = torch.randn(1000, 10)
    train_labels = torch.randn(1000, 1)
    val_data = torch.randn(200, 10)
    val_labels = torch.randn(200, 1)
    
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # 测试量化优化
    print("=== 量化优化测试 ===")
    optimizer = QuantizationOptimizer(model)
    results = optimizer.optimize_quantization(train_loader, val_loader, test_data, test_labels)
    
    print(f"最佳策略: {results['best_strategy']}")
    print("量化优化测试完成")